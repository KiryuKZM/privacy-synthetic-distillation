"""Loss-auditable public CityFM input representation; no labels, HTTP or models.

Coordinates are local metric offsets.  Independent public context is retained,
including enclosing campuses and indoor POIs; equivalent target footprints are
excluded as alternate records of the target. Source IDs occur only in the audit.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from privacy_synth.location_classification import SEMANTIC_KEYS

CONTEXT_KEYS = tuple(dict.fromkeys(('name', 'healthcare', *SEMANTIC_KEYS)))
PHYSICAL_KEYS = ('building:levels', 'height', 'min_height', 'building:min_level',
                 'roof:levels', 'roof:height')
REPRESENTATION_VERSION = 'cityfm-relative-public-context-v1'


def metric_polygon(record, transformer):
    """Project CityFM [latitude,longitude] exterior points without simplifying."""
    from shapely.geometry import Polygon
    from shapely import transform
    points = record.get('points', [])
    if len(points) < 3:
        raise ValueError('Insufficient polygon points')
    if any(len(p) != 2 or not (0 <= float(p[0]) <= 3 and
                               102 <= float(p[1]) <= 105) for p in points):
        raise ValueError('Invalid Singapore coordinate')
    polygon = transform(Polygon([(float(p[1]), float(p[0])) for p in points]),
                        transformer.transform, interleaved=False)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError('Invalid polygon')
    return polygon


def same_footprint(target_polygon, context_polygon):
    """Conservative geometry-equivalence test, not an overlap/site filter.

    Exact Shapely topological equality accommodates ring order and extra collinear
    vertices. No IoU threshold turns independent large site polygons into targets.
    """
    return target_polygon.equals(context_polygon)


def build_payload(target_id, target, listed_context_ids, entities, transformer):
    """Return (model payload, local-only provenance audit), using no gold table.

    The full listed native context is checked against a metric 100m radius.
    Invalid/missing associations fail closed rather than create silently sparse
    examples. Every retained neighbor has raw allowed semantics and relative
    position, regardless of nearest-neighbor rank.
    """
    from shapely.geometry import Point
    g = metric_polygon(target, transformer)
    x, y = transformer.transform(float(target['lon']), float(target['lat']))
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('Non-finite target location')
    footprint = [[round(px-x, 3), round(py-y, 3)] for px, py in g.exterior.coords]
    physical = {k: str(target['tags'][k]) for k in PHYSICAL_KEYS if k in target.get('tags', {})}
    geometry = {'area_m2': round(g.area, 1), 'perimeter_m': round(g.length, 1),
                'compactness': round(4*math.pi*g.area/g.length**2, 4),
                'bbox_width_m': round(g.bounds[2]-g.bounds[0], 1),
                'bbox_height_m': round(g.bounds[3]-g.bounds[1], 1),
                'vertex_count': len(g.exterior.coords)-1,
                'physical_tags': physical, 'footprint_xy_m': footprint}
    neighbors = []
    audit = {'retained': [], 'excluded': [], 'native_associations': len(listed_context_ids),
             'physical_fields': sorted(physical), 'raw_semantic_field_count': 0,
             'names_retained': 0, 'healthcare_retained': 0}
    seen = set()
    for raw_id in listed_context_ids:
        nid = str(raw_id)
        reason = None
        if nid in seen:
            reason = 'repeated_context_id'
        seen.add(nid)
        if nid == str(target_id):
            reason = 'same_target_id'
        if reason:
            audit['excluded'].append({'id': nid, 'reason': reason})
            continue
        if nid not in entities:
            raise ValueError(f'Missing context entity: {nid}')
        entity = entities[nid]
        nx, ny = transformer.transform(float(entity['lon']), float(entity['lat']))
        distance = math.hypot(nx-x, ny-y)
        if not math.isfinite(distance):
            raise ValueError(f'Invalid context location: {nid}')
        if distance > 100:
            audit['excluded'].append({'id': nid, 'reason': 'outside_100m', 'distance_m': distance})
            continue
        eg = metric_polygon(entity, transformer) if entity.get('type') == 'polygon' else None
        if eg is not None and same_footprint(g, eg):
            audit['excluded'].append({'id': nid, 'reason': 'equivalent_target_footprint'})
            continue
        if eg is not None:
            relation = ('encloses_target' if eg.covers(g) else 'inside_target' if g.covers(eg)
                        else 'overlaps_target' if eg.intersects(g) else 'nearby')
        else:
            relation = 'inside_target' if g.covers(Point(nx, ny)) else 'nearby'
        semantics = {k: str(entity['tags'][k]) for k in CONTEXT_KEYS if k in entity.get('tags', {})}
        # Empty semantics are represented explicitly rather than replaced by a
        # fabricated category. No raw address/contact/wiki/source IDs are copied.
        record = {'semantics': semantics, 'dx_m': round(nx-x, 1),
                  'dy_m': round(ny-y, 1), 'distance_m': round(distance, 1),
                  'relation': relation}
        neighbors.append((distance, nid, record))
        audit['retained'].append({'id': nid, 'relation': relation,
                                  'semantic_keys': sorted(semantics)})
        audit['raw_semantic_field_count'] += len(semantics)
        audit['names_retained'] += 'name' in semantics
        audit['healthcare_retained'] += 'healthcare' in semantics
    neighbors.sort(key=lambda n: (n[0], n[1]))
    records = [n[2] for n in neighbors]
    counts = Counter((k, v) for n in records for k, v in n['semantics'].items())
    payload = {'representation': REPRESENTATION_VERSION, 'target': geometry,
               'context_radius_m': 100, 'total_neighbors': len(records),
               'coordinate_frame': 'east/north metres relative to target; closed unsimplified footprint; coordinates rounded to 0.001m',
               'neighbor_columns': ['semantics', 'dx_m', 'dy_m', 'distance_m', 'relation'],
               'neighbors': [[n[k] for k in ('semantics', 'dx_m', 'dy_m', 'distance_m', 'relation')] for n in records],
               'all_neighbor_semantic_counts': [[k, v, c] for (k, v), c in sorted(counts.items())],
               'context_completeness': {'all_retained_semantics': True, 'all_retained_positions': True,
                                        'nearest_neighbor_truncation': False}}
    return payload, audit


def compact_payload(payload):
    """Lossless semantic-profile dictionary compression for long neighborhoods.

    Positions of *every* entity remain explicit; no nearest24 or dropped-name
    fallback. Consumers can reconstruct the expanded rows exactly.
    """
    if 'semantic_profiles' in payload:
        return payload
    profiles = []
    indexes = {}
    rows = []
    for semantics, dx, dy, distance, relation in payload['neighbors']:
        key = tuple(sorted(semantics.items()))
        if key not in indexes:
            indexes[key] = len(profiles)
            profiles.append(semantics)
        rows.append([indexes[key], dx, dy, distance, relation])
    return {**payload, 'semantic_profiles': profiles, 'neighbors': rows,
            'neighbor_columns': ['semantic_profile_index', 'dx_m', 'dy_m', 'distance_m', 'relation']}


def expand_neighbors(payload):
    """Decode either lossless context encoding for audits and local baselines."""
    if 'semantic_profiles' not in payload:
        return payload['neighbors']
    return [[payload['semantic_profiles'][i], dx, dy, d, relation]
            for i, dx, dy, d, relation in payload['neighbors']]


def repaired_baseline_features(payload):
    """Deterministic structured/name features, with vocabulary learned by caller.

    All target shape vertices, physical attributes and all neighbor semantics,
    distances and spatial relations contribute. Names use exact strings, word
    tokens and character 3--5grams; there is no externally learned vocabulary,
    label lookup, target ID or absolute position. Use a sparse DictVectorizer
    fitted on the fresh training partition, never on Dev/Screen/Sealed.
    """
    features = Counter()
    target = payload['target']
    for key, value in target.items():
        if isinstance(value, (int, float)):
            features['geometry:'+key] = float(value)
    for i, (x, y) in enumerate(target['footprint_xy_m']):
        features[f'footprint:{i}:x'] = x
        features[f'footprint:{i}:y'] = y
    for key, value in target['physical_tags'].items():
        features['physical:'+key+'='+value] += 1
        match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*', value)
        if match:
            features['physical_numeric:'+key] = float(match.group(1))
    for semantics, dx, dy, distance, relation in expand_neighbors(payload):
        features['relation:'+relation] += 1
        distance_bin = min(3, int(distance/25))
        for key, value in semantics.items():
            category = key+'='+value
            features['semantic:'+category] += 1
            features[f'bin:{distance_bin}:'+category] += 1
            features['relation_semantic:'+relation+':'+category] += 1
            features['east_sum:'+category] += dx
            features['north_sum:'+category] += dy
            if key == 'name':
                normalized = ' '.join(value.casefold().split())
                for word in re.findall(r'\w+', normalized):
                    features['name_word:'+word] += 1
                bounded = ' '+normalized+' '
                for width in (3, 4, 5):
                    for i in range(len(bounded)-width+1):
                        features['name_char:'+bounded[i:i+width]] += 1
    return dict(features)
