import geopandas as gpd
import json
import logging
import numpy as np
import requests
import secots
import shapely

from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection


LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
CONFIG_FILE = 'config.json'
LON_LAT_CRS = 'EPSG:4326'

DEFAULT_CONFIG = {
    'log_path': 'div4aep.log',
    'debug': False,
    'overpass_url': 'https://overpass-api.de/api/interpreter',
    'overpass_key': '',
    'overpass_timeout': 600,
    'max_patch_diameter': 1000000,
    'earth_radius': 6371000,
    'bbox': [-90, -180, 90, 180],
    'boundary_filter': '[type=boundary][boundary=administrative]',
    'initial_admin_level': 2,
    'max_admin_level': 6,
    'min_coverage': 0.7,
    'good_coverage': 0.9,
    'out_path': 'patches.csv',
    'max_overhang': 0.1,
    'patches_map_path': '',
    'uncovered_map_path': ''
}


logger = logging.getLogger(__name__)


def check_config(config):
    ''' Check config dict for missing keys (set default values) and for
    unknown keys. '''
    
    logger.debug('Checking config.')
    
    # missing keys
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
            logger.info(f'Key {repr(key)} missing in config, '
                        f'using default {repr(value)}.')
    
    # unknown keys
    keys = config.keys() - DEFAULT_CONFIG.keys()
    if len(keys) > 0:
        keys_str = ', '.join([repr(k) for k in keys])
        logger.warning(f'Unknown keys in config: {keys_str}.')


class OSMObject:

    def __init__(self, j, t):

        self.type = t
        self.id = j['id']
        if j.get('tags'):
            self.tags = j['tags']
        else:
            self.tags = dict()

    def __str__(self):

        return f'{self.type} {self.id}'

    def __repr__(self):

        return f'{self.type} {self.id}'


class Node(OSMObject):

    def __init__(self, j):

        super().__init__(j, 'node')
        self.lon = j['lon']
        self.lat = j['lat']


class Way(OSMObject):

    def __init__(self, j):

        super().__init__(j, 'way')
        self.node_ids = j['nodes']


class RelMember:

    def __init__(self, type_, id_, role):

        self.type = type_
        self.id = id_
        self.role = role


class Relation(OSMObject):

    def __init__(self, j):

        super().__init__(j, 'rel')
        self.members = [RelMember(m['type'], m['ref'], m['role']) for m in j['members']]


class Patch:

    @classmethod
    def _ways2shape(cls, ways, nodes_dict):
        lines = []
        for w in ways:
            points = [(nodes_dict[n_id].lon, nodes_dict[n_id].lat) for n_id in w.node_ids]
            lines.append(LineString(points))
        outline = shapely.line_merge(shapely.unary_union(lines))
        if isinstance(outline, LineString):
            outline = MultiLineString([outline])
        if isinstance(outline, MultiLineString):
            polys = []
            for line in outline.geoms:
                if len(line.coords) < 4 or line.coords[0] != line.coords[-1]:
                    ways_str = ', '.join([str(w.id) for w in ways])
                    logger.warning(
                        f'Could not create Polygon from merging ways {ways_str} to MultiLineString! '
                        f'LineString from {line.coords[0]} to {line.coords[-1]} is not a ring. '
                        f'Ignoring this line.'
                    )
                    continue
                polys.append(Polygon(line))
            shape = shapely.unary_union(polys)
        else:
            logger.error('Merging ways resulted in {type(outline)}, which is not supported!')
            shape = None
        return shape
    
    @classmethod
    def _rel2shape(cls, r, nodes_dict, ways_dict):
    
        # group members by type 'outer' or 'inner'
        outer_ways = []
        inner_ways = []
        for m in r.members:
            if m.role == 'outer' and m.type == 'way':
                outer_ways.append(ways_dict[m.id])
            elif m.role == 'inner' and m.type == 'way':
                inner_ways.append(ways_dict[m.id])
    
        # make shapes from way groups
        if outer_ways:
            outer_shape = cls._ways2shape(outer_ways, nodes_dict)
            if not outer_shape:
                return None
        else:
            logger.error('Relation {r.id} has no outer ways!')
            return None
        if not inner_ways:
            return outer_shape
        inner_shape = cls._ways2shape(inner_ways, nodes_dict)
        if not inner_shape:
            logger.warning('Could not create shape from inner ways. Ignoring inner rings.')
            return outer_shape
        try:
            shape = shapely.difference(outer_shape, inner_shape)
        except Exception as e:
            logger.warning(f'Relation {r.id} has invalid geometry ({e})! Ignoring inner rings.')
            shape = outer_shape

        return shape

    @classmethod
    def _shape2points(cls, shape):
    
        if isinstance(shape, Polygon):
            points = list(shape.exterior.coords)
        elif isinstance(shape, MultiPolygon) or isinstance(shape, GeometryCollection):
            points = []
            for subshape in shape.geoms:
                points.extend(cls._shape2points(subshape))
        else:
            logger.error(f'Cannot extract points from shape of type {type(shape)}!')
            points = []
                
        return points    

    @classmethod
    def _shape2bboxes(cls, shape):
        if isinstance(shape, Polygon):
            subshapes = [shape]
        elif isinstance(shape, MultiPolygon):
            subshapes = list(shape.geoms)
        else:
            logger.error(f'Cannot compute bboxes from shape of type {type(shape)}!')
            return []
        bboxes = []
        for s in subshapes:
            x, y = zip(*cls._shape2points(s))
            bboxes.append((min(y), min(x), max(y), max(x)))
        return bboxes

    @classmethod
    def _relative_area(cls, subshape, shape, lat, lon):
        shapes = gpd.GeoSeries([subshape, shape], crs=LON_LAT_CRS)
        shapes = shapes.to_crs(f'+proj=aeqd +lat_0={lat} +lon_0={lon}')
        return shapes.iloc[0].area / shapes.iloc[1].area
    
    @classmethod
    def from_subpatches_data(cls, nodes, ways, rels):

        nodes_dict = {n.id: n for n in nodes}
        ways_dict = {w.id: w for w in ways}

        patch = Patch(None, nodes_dict, ways_dict)
        patch.name = 'World'
        patch.code = 'WORLD'
        patch.osm_id = 0
        patch.admin_level = 0
        patch.coverage = None

        patch.subpatches = []
        for r in rels:
            subpatch = Patch(r, nodes_dict, ways_dict)
            if subpatch.name:
                patch.subpatches.append(subpatch)
            else:
                logger.warning(f'Ignoring relation {r.id} due to errors.')

        patch.shape = shapely.unary_union([subpatch.shape for subpatch in patch.subpatches])
        patch.points = patch._shape2points(patch.shape)
        patch.bboxes = patch._shape2bboxes(patch.shape)
        patch.lon = 0
        patch.lat = 0
        patch.radius = np.inf
        
        return patch
    

    def __init__(self, r, nodes_dict, ways_dict):

        if not r:  # root patch
            return
        
        logger.info(f'Creating patch from relation {r.id}...')

        # patch name
        self.name = r.tags.get('name')
        if not self.name:
            logger.error(f'Relation {r.id} does not have a name tag!')
            self.name = None
            return
        logger.debug(f'patch name: {self.name}')

        # patch code
        self.code = r.tags.get('ISO3166-1:alpha2') or r.tags.get('ISO3166-1:alpha3') or r.tags.get('ISO3166-2')
        if not self.code:  # make a 6-digit code by hashing all tag values
            self.code = str(hash(''.join(r.tags.values())))[-6:]
        logger.debug(f'patch code: {self.code}')

        # misc path properties
        self.osm_id = r.id
        self.admin_level = int(r.tags.get('admin_level'))
        self.coverage = None
        self.subpatches = []

        # get patch geometry
        self.shape = self._rel2shape(r, nodes_dict, ways_dict)
        if not self.shape:
            logger.error(f'Could not get shape for relation {r.id}!')
            self.name = None
            return
        self.points = self._shape2points(self.shape)
        if len(self.points) < 3:
            logger.error(f'Shape of relation {r.id} has only {len(self.points)} points!')
            self.name = None
            return
        self.bboxes = self._shape2bboxes(self.shape)
        if not self.bboxes:
            logger.error(f'Could not make bboxes for shape of relation {r.id}!')
            self.name = None
            return

        # get patch center and diameter
        try:
            self.lon, self.lat, self.radius = secots.smallest_circle(self.points)
        except secots.NotHemisphereError:
            self.lon, self.lat, self.radius = 0.0, 0.0, np.inf
        self.radius = self.radius * config['earth_radius']
        logger.debug(f'patch diameter: {2 * self.radius}')

        logger.info('...done.')

        # do we have to split the patch?
        if 2 * self.radius <= config['max_patch_diameter']:
            return
        logger.info(f'Splitting patch "{self.name}" (relation {self.osm_id})...')

        # find subpatches
        best_coverage = None
        best_subpatches = []
        subpatches = []
        prev_subpatches = []
        for admin_level in range(self.admin_level + 1, config['max_admin_level'] + 1):
            logger.info(f'Looking for subpatches at admin_level {admin_level}...')

            # query relations for current admin_level
            bbox_union = '('
            for bbox in self.bboxes:
                bbox_str = '(' + ','.join([str(coord) for coord in bbox]) + ')'
                bbox_union += f'rel{config["boundary_filter"]}[admin_level={admin_level}]{bbox_str};'
            bbox_union += ')'
            query = f'{bbox_union};\n (._;>;);\n out;'
            nodes, ways, rels = overpass(query)
            if len(rels) == 0:
                logger.info('...nothing found.')
                continue
            logger.info(f'...found {len(rels)} relations.')

            # make subpatches
            nodes_dict = {n.id: n for n in nodes}
            ways_dict = {w.id: w for w in ways}
            for r in rels:
                subshape = self._rel2shape(r, nodes_dict, ways_dict)
                if not subshape:
                    logger.warning(f'Ignoring relation {r.id} due to errors.')
                    continue
                overhang = shapely.difference(subshape, self.shape)
                if self._relative_area(overhang, subshape, self.lat, self.lon) > config['max_overhang']:
                #if not shapely.contains(self.shape, subshape):
                    logger.debug(f'Ignoring relation {r.id}, because it\'s not contained in parent.')
                    continue
                subpatches.append(Patch(r, nodes_dict, ways_dict))
            logger.info(f'There\'re {len(subpatches)} subpatches at admin_level {admin_level}.')
            if len(subpatches) == 0:
                continue
            
            # check whether subpatches from previous (non-empty) admin_level should we kept
            shape = shapely.union_all([subpatch.shape for subpatch in subpatches])
            prev_shape = shapely.union_all([subpatch.shape for subpatch in prev_subpatches])
            if prev_subpatches and shapely.intersection(shape, prev_shape).area == 0:
                logger.debug(f'Joining with subpatches from previous admin_level.')
                shape = shapely.union_all([shape, prev_shape])
                subpatches.extend(prev_subpatches)

            # check coverage
            diff = shapely.difference(self.shape, shape)
            if diff.is_empty:
                logger.info('Full coverage.')
                self.coverage = 1
                self.subpatches = subpatches
                break
            coverage = 1 - self._relative_area(diff, self.shape, self.lat, self.lon)
            if coverage >= config['good_coverage']:
                logger.info(f'Good coverage ({coverage}).')
                self.coverage = coverage
                self.subpatches = subpatches
                break
            if not best_coverage or coverage > best_coverage:
                best_coverage = coverage
                best_subpatches = subpatches
            logger.debug(f'Current coverage: {coverage}, best coverage: {best_coverage}.')

            prev_subpatches = subpatches
            subpatches = []

        else:
            if best_coverage and best_coverage >= config['min_coverage']:
                logger.info(f'No more admin levels. Using best coverage result so far ({best_coverage}).')
                self.coverage = best_coverage
                self.subpatches = best_subpatches
            else:
                logger.info('No more admin levels. No subpatchess with sufficient coverage found.')
        
        logger.info(f'...finished splitting patch {self.name} (relation {self.osm_id}).')
    
    
    def __str__(self):
        return f'Patch ("{self.name}", "{self.code}", {self.osm_id}, admin_level={self.admin_level})'
        

def filesize2str(size):
    '''
    Convert integer to human readable file size string.
    '''

    if size < 1000:
        return f'{size} byte'
    elif size < 1000 ** 2:
        return f'{size / 1000:.0f} kB'
    elif size < 1000 ** 3:
        return f'{size / (1000 ** 2):.0f} MB'
    else:
        return f'{size / (1000 ** 3):.0f} GB'
    

def overpass(query, ids_only=False):
    '''
    Send query to Overpass API and parse results to Python objects.

    :param str query: Query string without config options like [timeout: ...].
    :param bool ids_only: If True, return OSM IDs instead of Python objects.
    :return: lists of nodes, ways, relations (objects or IDs).
    :rtype: ([Node], [Way], [Relation])
    '''

    # get data from Overpass API
    logger.info('Sending query to Overpass API...')
    preamble = f'[output: json][timeout: {config["overpass_timeout"]}];\n'
    response = requests.post(
        config['overpass_url'],
        data={'data': preamble + query},
        headers={'X-API-Key': config['overpass_key']}
    )
    if response.status_code != 200:
        logger.error(
            f'Overpass server returned {response.status_code}. Query was:\n'
            f'{preamble + query}'
        )
        return [], [], []
    size_str = filesize2str(len(response.content))
    logger.info(f'...received {size_str} from Overpass API.')

    # parse JSON
    j = response.json()
    objects = j['elements']
    if j.get('remarks'):
        logger.debug(f'Overpass API sent a remark: {j["remarks"]}')
    #if len(objects) == 0:
    #    logger.debug(f'Overpass did not return any objects.'
    #                 f'Returned content: {response.content.decode()}')

    # make objects (or IDs)
    if ids_only:
        nodes = [o.get('id') for o in objects if o.get('type') == 'node']
        ways = [o.get('id') for o in objects if o.get('type') == 'way']
        rels = [o.get('id') for o in objects if o.get('type') == 'relation']
    else:
        nodes = [Node(o) for o in objects if o.get('type') == 'node']
        ways = [Way(o) for o in objects if o.get('type') == 'way']
        rels = [Relation(o) for o in objects if o.get('type') == 'relation']
    other = len(objects) - len(nodes) - len(ways) - len(rels)
    logger.debug(f'Found {len(objects)} objects, {len(nodes)} nodes, '
                 f'{len(ways)} ways, {len(rels)} rels, {other} other objects.')
        
    return nodes, ways, rels


def main():

    global config, logger

    # set up logging to console
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('Set log level to INFO.')
    
    # load config file
    logger.info(f'Reading config file {CONFIG_FILE}.')
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        del f
    except Exception as e:
        logger.error(f'Could not load config file config.json ({e}).')
        return
    check_config(config)

    # set up logging to log file
    logger.info(f'Starting logging to file {config['log_path']}.')
    try:
        handler = logging.FileHandler(config['log_path'], mode='w')
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    except Exception as e:
        logger.error(f'Creating log file failed ({e}).')

    # set log level
    if config.get('debug', False):
        logger.setLevel(logging.DEBUG)
        logger.info('Set log level to DEBUG.')

    # show major config items
    logger.info(f'Using Overpass API at {config["overpass_url"]}.')
    logger.info(f'Maximum patch diameter is {config["max_patch_diameter"]} meters.')
    logger.info(f'Earth radius is {config["earth_radius"]} meters.')

    # get all admin_level=2 relations
    bbox_str = '(' + ','.join([str(f) for f in config['bbox']]) + ')'
    query = (
        f'rel{config["boundary_filter"]}'
        f'   [admin_level={config["initial_admin_level"]}]'
        f'   {bbox_str};\n'
        f'(._;>;);\n'
        f'out;'
    )
    nodes, ways, rels = overpass(query)

    # create root patch
    global root_patch
    root_patch = Patch.from_subpatches_data(nodes, ways, rels)

    # get leaf patches
    leaf_patches = []
    leaf_parent_ids = []
    patches = [(0, p) for p in root_patch.subpatches]
    while len(patches) > 0:
        parent_id, patch = patches.pop()
        if patch.subpatches:
            patches.extend([(patch.osm_id, p) for p in patch.subpatches])
        else:
            leaf_patches.append(patch)
            leaf_parent_ids.append(parent_id)
    
    # write CSV file
    names = []
    codes = []
    osm_ids = []
    admin_levels = []
    lons = []
    lats = []
    radiuses = []
    for patch in leaf_patches:
        names.append(patch.name)
        codes.append(patch.code)
        osm_ids.append(patch.osm_id)
        admin_levels.append(patch.admin_level)
        lons.append(patch.lon)
        lats.append(patch.lat)
        radiuses.append(patch.radius)
    patches = gpd.pd.DataFrame(
        data={'osm_id': osm_ids, 'name': names, 'code': codes, 'admin_level': admin_levels,
              'parent_osm_id': leaf_parent_ids, 'lon': lons, 'lat': lats, 'radius': radiuses}
    )
    patches.to_csv(config['out_path'], index=False)
    
    # make maps
    shapes = [p.shape for p in leaf_patches]
    all_patches_shape = shapely.unary_union(shapes)
    diff = root_patch.shape - all_patches_shape
    if config['uncovered_map_path']:
        gpd.GeoSeries(diff, crs=LON_LAT_CRS).explore().save(config['uncovered_map_path'])
    if config['patches_map_path']:
        gpd.GeoSeries(shapes, crs=LON_LAT_CRS).explore().save(config['patches_map_path'])    


if __name__ == '__main__':
    main()
