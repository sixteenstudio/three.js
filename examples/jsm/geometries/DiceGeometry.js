import {
	BufferGeometry,
	Matrix4,
	SphereGeometry,
	Vector3
} from 'three';
import { RoundedBoxGeometry } from '../geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from '../utils/BufferGeometryUtils.js';

const _normal = new Vector3();
const _tangent = new Vector3();
const _bitangent = new Vector3();
const _center = new Vector3();
const _matrix = new Matrix4();

const PIP_LAYOUTS = {
	1: [ [ 0, 0 ] ],
	2: [ [ - 0.45, - 0.45 ], [ 0.45, 0.45 ] ],
	3: [ [ - 0.45, - 0.45 ], [ 0, 0 ], [ 0.45, 0.45 ] ],
	4: [ [ - 0.45, - 0.45 ], [ 0.45, - 0.45 ], [ - 0.45, 0.45 ], [ 0.45, 0.45 ] ],
	5: [ [ - 0.45, - 0.45 ], [ 0.45, - 0.45 ], [ 0, 0 ], [ - 0.45, 0.45 ], [ 0.45, 0.45 ] ],
	6: [ [ - 0.45, - 0.65 ], [ - 0.45, 0 ], [ - 0.45, 0.65 ], [ 0.45, - 0.65 ], [ 0.45, 0 ], [ 0.45, 0.65 ] ]
};

const FACES = [
	{ value: 1, normal: [ 0, 0, 1 ], up: [ 0, 1, 0 ] },
	{ value: 6, normal: [ 0, 0, - 1 ], up: [ 0, 1, 0 ] },
	{ value: 5, normal: [ 0, 1, 0 ], up: [ 0, 0, - 1 ] },
	{ value: 2, normal: [ 0, - 1, 0 ], up: [ 0, 0, 1 ] },
	{ value: 3, normal: [ 1, 0, 0 ], up: [ 0, 1, 0 ] },
	{ value: 4, normal: [ - 1, 0, 0 ], up: [ 0, 1, 0 ] }
];

function getFaceBasis( normal, up ) {

	_normal.set( ...normal ).normalize();
	_tangent.crossVectors( _normal, new Vector3( ...up ).normalize() ).normalize();
	_bitangent.crossVectors( _normal, _tangent ).normalize();

	return {
		normal: _normal.clone(),
		tangent: _tangent.clone(),
		bitangent: _bitangent.clone()
	};

}

function createPipGeometry( center, radius, segments = 16 ) {

	const geometry = new SphereGeometry( radius, segments, segments ).toNonIndexed();
	geometry.applyMatrix4( _matrix.makeTranslation( center.x, center.y, center.z ) );
	return geometry;

}

function createLoadedPlugGeometry( face, size, half, recess ) {

	const { normal, tangent, bitangent } = getFaceBasis( face.normal, face.up );
	const plugRadius = size * 0.18;
	const depth = recess * 2.5;

	_center.copy( normal ).multiplyScalar( half - depth );
	const geometry = new SphereGeometry( plugRadius, 24, 16 ).toNonIndexed();
	geometry.scale( 1, 0.35, 1 );
	geometry.applyMatrix4( _matrix.makeBasis( tangent, bitangent, normal ) );
	geometry.applyMatrix4( _matrix.makeTranslation( _center.x, _center.y, _center.z ) );

	return geometry;

}

/**
 * Procedural rounded die geometry with recessed gemstone pips.
 *
 * @param {Object} [parameters]
 * @param {number} [parameters.size=1] - Edge length of the die.
 * @param {number} [parameters.segments=6] - Rounded corner segments.
 * @param {number} [parameters.radius=0.12] - Corner rounding radius relative to unit cube.
 * @param {number} [parameters.pipRadius=0.055] - Pip radius relative to unit cube.
 * @param {number} [parameters.recess=0.03] - Pip recess depth.
 * @param {boolean} [parameters.pips=true] - Whether to generate pip geometry.
 * @param {boolean} [parameters.loadedPlug=false] - Replace the 2-face pips with a weighted plug.
 * @returns {BufferGeometry}
 */
function createDiceGeometry( parameters = {} ) {

	const size = parameters.size !== undefined ? parameters.size : 1;
	const segments = parameters.segments !== undefined ? parameters.segments : 6;
	const radius = parameters.radius !== undefined ? parameters.radius : 0.12;
	const pipRadius = parameters.pipRadius !== undefined ? parameters.pipRadius : 0.055;
	const recess = parameters.recess !== undefined ? parameters.recess : 0.03;
	const pips = parameters.pips !== undefined ? parameters.pips : true;
	const loadedPlug = parameters.loadedPlug !== undefined ? parameters.loadedPlug : false;

	const half = size * 0.5;
	const scale = size;

	const body = new RoundedBoxGeometry( scale, scale, scale, segments, radius * scale );
	const pipGeometries = [];
	let plugGeometry = null;
	const pipScale = size;

	if ( pips ) {

		for ( const face of FACES ) {

			const { normal, tangent, bitangent } = getFaceBasis( face.normal, face.up );

			if ( loadedPlug && face.value === 2 ) {

				plugGeometry = createLoadedPlugGeometry( face, pipScale, half, recess );
				continue;

			}

			const layout = PIP_LAYOUTS[ face.value ];

			for ( const [ u, v ] of layout ) {

				_center.copy( normal ).multiplyScalar( half - recess );
				_center.addScaledVector( tangent, u * pipScale * 0.34 );
				_center.addScaledVector( bitangent, v * pipScale * 0.34 );

				pipGeometries.push( createPipGeometry( _center, pipRadius * pipScale ) );

			}

		}

	}

	const parts = [ body ];

	if ( pipGeometries.length > 0 ) {

		parts.push( mergeGeometries( pipGeometries ) );

	}

	if ( plugGeometry ) {

		parts.push( plugGeometry );

	}

	const geometry = mergeGeometries( parts, true );
	geometry.computeVertexNormals();

	return geometry;

}

export { createDiceGeometry, FACES, PIP_LAYOUTS };
