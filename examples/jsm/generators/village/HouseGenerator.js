import {
	BoxGeometry,
	BufferAttribute,
	BufferGeometry,
	ExtrudeGeometry,
	InterpolationSamplingMode,
	InterpolationSamplingType,
	Matrix3,
	Matrix4,
	Mesh,
	MeshStandardMaterial,
	Path,
	PlaneGeometry,
	Shape,
	Sphere,
	Vector2,
	Vector3
} from 'three';

import { MeshStandardNodeMaterial } from 'three/webgpu';
import { attribute, cameraPosition, color, cross, dot, float, floor, Fn, fract, fwidth, hash as ihash, mix, modelWorldMatrixInverse, normalLocal, normalWorldGeometry, positionLocal, positionView, positionWorld, normalView, select, smoothstep, step, uint, uv, varying, vec2, vec3, vec4 } from 'three/tsl';

const _scale = /*@__PURE__*/ new Vector3();
const _position = /*@__PURE__*/ new Vector3();
const _matrix = /*@__PURE__*/ new Matrix4();
const _normalMatrix = /*@__PURE__*/ new Matrix3();
const _identity = /*@__PURE__*/ new Matrix4();

// material-zone codes baked per vertex into the merged geometry, so one material can
// branch on partId and shade every zone. BRICK is the solid fired clay of ridge
// caps, verge tiles and chimney stacks — tile-toned, but without the slope's
// barrel pattern, which would stretch with each box's size.
const PartId = { WALL: 0, TRIM: 1, SHUTTER: 2, ROOF: 3, GLASS: 4, DOOR: 5, STONE: 6, RAIL: 7, TERRACE: 8, BRICK: 9 };
const { WALL, TRIM, SHUTTER, ROOF, GLASS, DOOR, STONE, RAIL, TERRACE, BRICK } = PartId;

// the flat painted band around every opening; the sills, cornices and fascias key
// off it so all the facade trim reads as one consistent painted system
const SURROUND = 0.09;

// merging requires all-indexed or all-non-indexed inputs; extrusions are
// non-indexed while boxes/planes are indexed, so normalize before merging

function nonIndexed( geometry ) {

	return geometry.index ? geometry.toNonIndexed() : geometry;

}

// the unit box is identical for every house's boxes — build it once
const _unitBox = /*@__PURE__*/ nonIndexed( new BoxGeometry( 1, 1, 1 ) );

/**
 * Bakes a list of instance groups into one non-indexed BufferGeometry. Each group is a
 * base geometry ( position + normal + uv ), an array of Matrix4 placements and a `partId`
 * written to a per-vertex attribute, together with the constant per-house `paint`
 * ( palette index ) and `houseId` ( 0..1 hash ) attributes the shared material reads.
 * A glass group also carries `rooms` — the interior-mapping room ( centre + size ) each
 * pane looks into, baked per vertex so the material can shade a furnished interior.
 * Transforming straight into preallocated typed arrays avoids mergeGeometries'
 * per-instance allocations; the result is one geometry, ready for a single draw call.
 */
function bakeGroups( groups, paint = 0, houseId = 0 ) {

	let total = 0;
	for ( const group of groups ) total += group.geometry.attributes.position.count * group.matrices.length;

	const position = new Float32Array( total * 3 );
	const normal = new Float32Array( total * 3 );
	const uv = new Float32Array( total * 2 );
	const partId = new Float32Array( total );
	const paints = new Float32Array( total ).fill( paint );
	const houseIds = new Float32Array( total ).fill( houseId );
	const roomCenter = new Float32Array( total * 3 );
	const roomSize = new Float32Array( total * 2 );

	let w = 0;

	// the bounding sphere falls out of the AABB gathered while transforming, sparing a
	// second full pass over the positions ( computeBoundingSphere )
	let minX = Infinity, minY = Infinity, minZ = Infinity;
	let maxX = - Infinity, maxY = - Infinity, maxZ = - Infinity;

	for ( const group of groups ) {

		const geometry = group.geometry;
		const P = geometry.attributes.position.array;
		const N = geometry.attributes.normal.array;
		const U = geometry.attributes.uv.array;
		const count = geometry.attributes.position.count;
		const id = group.partId;
		const rooms = group.rooms; // per-instance { center, size }, glass only
		const rigid = group.rigid === true; // pure rotation ( + translation ): the normal matrix is the rotation itself

		for ( let i = 0; i < group.matrices.length; i ++ ) {

			const room = rooms ? rooms[ i ] : null;
			const matrix = group.matrices[ i ];
			const e = matrix.elements;
			const e0 = e[ 0 ], e1 = e[ 1 ], e2 = e[ 2 ], e4 = e[ 4 ], e5 = e[ 5 ], e6 = e[ 6 ], e8 = e[ 8 ], e9 = e[ 9 ], e10 = e[ 10 ], e12 = e[ 12 ], e13 = e[ 13 ], e14 = e[ 14 ];

			// for a rigid frame the inverse-transpose equals the rotation, so its columns
			// are read straight from the matrix and the per-instance 3×3 inverse is skipped
			let n0, n1, n2, n3, n4, n5, n6, n7, n8;

			if ( rigid ) {

				n0 = e0; n1 = e1; n2 = e2; n3 = e4; n4 = e5; n5 = e6; n6 = e8; n7 = e9; n8 = e10;

			} else {

				const ne = _normalMatrix.getNormalMatrix( matrix ).elements;
				n0 = ne[ 0 ]; n1 = ne[ 1 ]; n2 = ne[ 2 ]; n3 = ne[ 3 ]; n4 = ne[ 4 ]; n5 = ne[ 5 ]; n6 = ne[ 6 ]; n7 = ne[ 7 ]; n8 = ne[ 8 ];

			}

			for ( let v = 0; v < count; v ++ ) {

				const v3 = v * 3, w3 = w * 3;
				const x = P[ v3 ], y = P[ v3 + 1 ], z = P[ v3 + 2 ];
				const wx = e0 * x + e4 * y + e8 * z + e12;
				const wy = e1 * x + e5 * y + e9 * z + e13;
				const wz = e2 * x + e6 * y + e10 * z + e14;
				position[ w3 ] = wx; position[ w3 + 1 ] = wy; position[ w3 + 2 ] = wz;
				if ( wx < minX ) minX = wx; if ( wx > maxX ) maxX = wx;
				if ( wy < minY ) minY = wy; if ( wy > maxY ) maxY = wy;
				if ( wz < minZ ) minZ = wz; if ( wz > maxZ ) maxZ = wz;

				const nx = N[ v3 ], ny = N[ v3 + 1 ], nz = N[ v3 + 2 ];
				const tx = n0 * nx + n3 * ny + n6 * nz, ty = n1 * nx + n4 * ny + n7 * nz, tz = n2 * nx + n5 * ny + n8 * nz;
				const inv = 1 / ( Math.sqrt( tx * tx + ty * ty + tz * tz ) || 1 );
				normal[ w3 ] = tx * inv; normal[ w3 + 1 ] = ty * inv; normal[ w3 + 2 ] = tz * inv;

				uv[ w * 2 ] = U[ v * 2 ]; uv[ w * 2 + 1 ] = U[ v * 2 + 1 ];
				partId[ w ] = id;

				if ( room !== null ) {

					roomCenter[ w3 ] = room.center.x; roomCenter[ w3 + 1 ] = room.center.y; roomCenter[ w3 + 2 ] = room.center.z;
					roomSize[ w * 2 ] = room.size.x; roomSize[ w * 2 + 1 ] = room.size.y;

				}

				w ++;

			}

		}

	}

	const geometry = new BufferGeometry();
	geometry.setAttribute( 'position', new BufferAttribute( position, 3 ) );
	geometry.setAttribute( 'normal', new BufferAttribute( normal, 3 ) );
	geometry.setAttribute( 'uv', new BufferAttribute( uv, 2 ) );
	geometry.setAttribute( 'partId', new BufferAttribute( partId, 1 ) );
	geometry.setAttribute( 'paint', new BufferAttribute( paints, 1 ) );
	geometry.setAttribute( 'houseId', new BufferAttribute( houseIds, 1 ) );
	geometry.setAttribute( 'roomCenter', new BufferAttribute( roomCenter, 3 ) );
	geometry.setAttribute( 'roomSize', new BufferAttribute( roomSize, 2 ) );

	geometry.boundingSphere = new Sphere(
		new Vector3( ( minX + maxX ) / 2, ( minY + maxY ) / 2, ( minZ + maxZ ) / 2 ),
		Math.hypot( maxX - minX, maxY - minY, maxZ - minZ ) / 2
	);

	return geometry;

}

// deterministic PRNG ( mulberry32 ) so a given seed always yields the same house

function createRandom( seed ) {

	let s = ( seed >>> 0 ) || 1;

	return function () {

		s = ( s + 0x6D2B79F5 ) | 0;
		let t = Math.imul( s ^ ( s >>> 15 ), 1 | s );
		t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

// an axis-aligned box matrix: the shared unit box scaled to size, centred at ( x, y, z ),
// optionally spun about Y — most of a house's trim, shutters and rails place this way
function boxMatrix( x, y, z, sizeX, sizeY, sizeZ, rotationY = 0 ) {

	_matrix.makeRotationY( rotationY );

	return new Matrix4()
		.copy( _matrix )
		.scale( _scale.set( sizeX, sizeY, sizeZ ) )
		.setPosition( _position.set( x, y, z ) );

}

// rescales a geometry's 0..1 UVs to metres, so the material's tile / plank
// patterns key off real-world sizes on every house
function metricUV( geometry, scaleU, scaleV ) {

	const uv = geometry.attributes.uv;

	for ( let i = 0; i < uv.count; i ++ ) {

		uv.setXY( i, uv.getX( i ) * scaleU, uv.getY( i ) * scaleV );

	}

	return geometry;

}

/**
 * Generates the narrow, colour-washed row houses of the Ligurian riviera — the
 * stacked facades of the Cinque Terre — from a small set of parameters.
 *
 * A house is a plastered shell: front and back walls are extruded with their
 * window and door openings punched through ( so every opening has real depth ),
 * the side walls carry the gable, and the top is capped per seed by a
 * low-pitched terracotta gable roof, a flat cotto-tiled terrace behind a
 * parapet, or a combination — a gable sharing the top with a railed balcony
 * terrace. Each opening is dressed with a painted surround, a sill, recessed
 * dark glazing and a pair of louvered shutters — folded open against the wall
 * or closed over the window — and some upper openings become French doors
 * behind small iron balconies. A stone podium below the ground floor lets the
 * house sit on sloping ground and reads as the retaining wall of its terrace.
 *
 * Everything is baked — via {@link bakeGroups} — into a single non-indexed
 * BufferGeometry tagged with a per-vertex `partId` ({@link PartId}) plus the
 * house's palette pick, so one material dresses a whole village.
 *
 * ```js
 * const generator = new HouseGenerator( { seed: 12, floors: 3 }, material );
 * scene.add( generator.build() ); // a single Mesh
 * ```
 */
class HouseGenerator {

	constructor( parameters = {}, material = null ) {

		this.parameters = parameters; // caller overrides; defaults + seed fill the rest at build time
		this.material = material; // a single material; the look is driven by the baked `partId` attribute

		this.mesh = null;

	}

	build() {

		const random = createRandom( this.parameters.seed ?? HouseGenerator.defaults.seed );

		// precedence: fixed defaults < seed-driven style < caller parameters

		const p = Object.assign( {}, HouseGenerator.defaults, randomStyle( random ), this.parameters );

		const w = p.width;
		const d = p.depth;
		const t = 0.24; // wall thickness — enough for the openings to read as real reveals

		const eaves = p.floors * p.floorHeight + 0.15;

		// the roof plan: a full gable, a flat terrace behind a parapet, or a
		// gable sharing the top with a terrace ( the balcony half )

		let gableSpan = null;
		let terraceSpan = null;

		if ( p.roofStyle === 'flat' ) {

			terraceSpan = [ - w / 2, w / 2 ];

		} else if ( p.roofStyle === 'combo' && w > 6.2 ) {

			const split = p.terraceSide * ( w / 2 - w * p.terraceFraction );
			gableSpan = p.terraceSide < 0 ? [ split, w / 2 ] : [ - w / 2, split ];
			terraceSpan = p.terraceSide < 0 ? [ - w / 2, split ] : [ split, w / 2 ];

		} else {

			gableSpan = [ - w / 2, w / 2 ];

		}

		const rise = ( d / 2 ) * p.roofPitch;
		const ridge = eaves + ( gableSpan ? rise : 0 );

		// one accumulator per kind of part: extruded walls in house space, plus
		// instance matrices for every repeated box / plane module

		const extras = []; // extruded walls and roof slopes ( { geometry, partId } ), already in house-local space
		const trim = []; // surrounds, sills, cornices, fascias, chimney caps ( unit boxes )
		const shutters = [];
		const rails = [];
		const stone = []; // podium and door steps
		const brick = []; // ridge capping, verge tiles and chimney stacks — solid fired clay
		const terraceBoxes = []; // flat cotto terrace floors
		const wallBoxes = []; // parapets
		const glass = []; // window panes ( unit planes )
		const glassRooms = []; // per-pane interior-mapping room ( centre + size ), aligned with `glass`
		const doors = [];

		const modules = { trim, shutters, rails, stone, glass, glassRooms, doors };

		// front and back facades; the back is plainer ( fewer, smaller openings ),
		// as it usually faces the hill or the next terrace up

		buildFacade( extras, modules, w, d, t, eaves, p, random, 0, true );
		buildFacade( extras, modules, w, d, t, eaves, p, random, Math.PI, p.backOpenings );

		// side walls carry the gable where a slope meets them: a pentagon profile
		// extruded to the wall thickness, inset so attached neighbours' walls
		// never sit coplanar; under a terrace the profile stays flat-topped

		for ( const side of [ - 1, 1 ] ) {

			const exposed = side === - 1 ? p.leftExposed : p.rightExposed;
			const gabled = gableSpan !== null && ( side < 0 ? gableSpan[ 0 ] < - w / 2 + 0.01 : gableSpan[ 1 ] > w / 2 - 0.01 );
			buildGableWall( extras, modules, w, d, t, eaves, ridge, p, random, side, exposed, gabled );

		}

		// the roof: tiled slopes with overhanging eaves over the gabled span, a
		// walkable terrace behind a parapet over the rest

		if ( gableSpan ) buildRoof( extras, trim, brick, w, d, eaves, ridge, p, gableSpan );
		if ( terraceSpan ) buildTerrace( wallBoxes, trim, rails, terraceBoxes, w, d, eaves, terraceSpan, p.roofStyle === 'combo' );

		// chimneys: short brick stacks with a cap slab, straddling the ridge

		const chimneys = gableSpan && gableSpan[ 1 ] - gableSpan[ 0 ] > 2.4 ? Math.floor( random() * 2.4 ) : 0; // 0..2, most houses have at least one

		for ( let i = 0; i < chimneys; i ++ ) {

			const cx = gableSpan[ 0 ] + 0.8 + random() * ( gableSpan[ 1 ] - gableSpan[ 0 ] - 1.6 );
			const cz = ( random() - 0.5 ) * 1.2;
			const top = ridge + 0.4 + random() * 0.5;
			const size = 0.4 + random() * 0.15;

			brick.push( boxMatrix( cx, ( top + ridge - 0.8 ) / 2, cz, size, top - ridge + 0.8, size ) );
			trim.push( boxMatrix( cx, top + 0.04, cz, size + 0.14, 0.08, size + 0.14 ) );

		}

		// the stone podium: sunk below the ground floor and slightly proud of the
		// walls, it ties the house to sloping ground and reads as the terrace's
		// retaining wall where the hill falls away in front

		if ( p.podiumDepth > 0 ) {

			stone.push( boxMatrix( 0, - p.podiumDepth / 2 + 0.06, 0, w + 0.08, p.podiumDepth + 0.12, d + 0.08 ) );

		}

		// --- bake every part into one geometry ---------------------------------

		// one mesh = one draw the renderer can't sort, so bake order is draw order:
		// the opaque facade first, the recessed glazing last

		const groups = [];

		for ( const extra of extras ) groups.push( { geometry: nonIndexed( extra.geometry ), matrices: [ _identity ], partId: extra.partId, rigid: true } );

		if ( trim.length > 0 ) groups.push( { geometry: _unitBox, matrices: trim, partId: TRIM } );
		if ( shutters.length > 0 ) groups.push( { geometry: _unitBox, matrices: shutters, partId: SHUTTER } );
		if ( wallBoxes.length > 0 ) groups.push( { geometry: _unitBox, matrices: wallBoxes, partId: WALL } );
		if ( brick.length > 0 ) groups.push( { geometry: _unitBox, matrices: brick, partId: BRICK } );
		if ( terraceBoxes.length > 0 ) groups.push( { geometry: _unitBox, matrices: terraceBoxes, partId: TERRACE } );
		if ( rails.length > 0 ) groups.push( { geometry: _unitBox, matrices: rails, partId: RAIL } );
		if ( stone.length > 0 ) groups.push( { geometry: _unitBox, matrices: stone, partId: STONE } );
		if ( doors.length > 0 ) groups.push( { geometry: nonIndexed( new PlaneGeometry( 1, 1 ) ), matrices: doors, partId: DOOR } );
		if ( glass.length > 0 ) groups.push( { geometry: nonIndexed( new PlaneGeometry( 1, 1 ) ), matrices: glass, partId: GLASS, rooms: glassRooms } );

		const paint = p.paint >= 0 ? p.paint : Math.floor( random() * housePalette.length );
		const geometry = bakeGroups( groups, paint, random() );

		const mesh = new Mesh( geometry, this.material || new MeshStandardMaterial( { color: 0xd9a05a, roughness: 0.9 } ) );
		mesh.name = 'House';

		this.dispose();
		this.mesh = mesh;

		return mesh;

	}

	dispose() {

		if ( this.mesh === null ) return;

		this.mesh.geometry.dispose();
		this.mesh = null;

	}

}

// fixed baseline. the remaining parameters ( openings, roof pitch, shutter odds )
// are derived from the seed by randomStyle() unless the caller provides them.
HouseGenerator.defaults = {
	seed: 1,
	width: 7,
	depth: 9,
	floors: 3,
	floorHeight: 3,
	paint: - 1, // housePalette index; -1 picks from the seed
	podiumDepth: 8, // how far the stone base runs below y = 0, to meet falling ground
	backOpenings: true,
	leftExposed: false, // exposed side walls ( row ends ) get windows of their own
	rightExposed: false,
	balconyChance: 0.22,
	closedChance: 0.3 // fraction of shutters closed over their window
};

// the seed-driven "style" of a house: bay rhythm, opening sizes and roof pitch.
// these sit between the fixed defaults and the caller's parameters, so any
// parameter passed in still overrides its seeded value.

function randomStyle( random ) {

	const roof = random();

	return {
		bayPitch: 1.75 + random() * 0.45,
		openingWidth: 0.95 + random() * 0.18,
		openingHeight: 1.38 + random() * 0.2,
		roofPitch: 0.38 + random() * 0.14, // rise / run — the low gables of the coast
		roofStyle: roof < 0.55 ? 'gable' : ( roof < 0.75 ? 'flat' : 'combo' ), // the mixed roofscape of the coast
		terraceFraction: 0.38 + random() * 0.22, // how much of a combo roof the terrace takes
		terraceSide: random() < 0.5 ? - 1 : 1
	};

}

// --- walls -----------------------------------------------------------------

// lays out one facade: the wall slab extruded with its openings punched through,
// then a dressed window or door module in each
function buildFacade( extras, modules, w, d, t, eaves, p, random, rotationY, withOpenings ) {

	const inset = 0.008; // a hair's width in from the party wall, so attached facades meet at a crisp seam
	const halfW = w / 2 - inset;

	const shape = new Shape();
	shape.moveTo( - halfW, 0 );
	shape.lineTo( halfW, 0 );
	shape.lineTo( halfW, eaves );
	shape.lineTo( - halfW, eaves );
	shape.lineTo( - halfW, 0 );

	const openings = [];

	if ( withOpenings ) {

		const back = rotationY !== 0;

		// bays: openings stack in vertical ranks, the Ligurian rhythm. margins keep
		// the end bays clear of the party walls.
		const margin = 0.55;
		const bays = Math.max( 1, Math.min( 6, Math.round( ( w - margin * 2 ) / p.bayPitch ) ) );
		const pitch = ( w - margin * 2 ) / bays;

		const ow = Math.min( p.openingWidth, pitch - 0.5 );
		const oh = p.openingHeight;

		// the front door lands in an end bay more often than not
		const doorBay = random() < 0.6 ? ( random() < 0.5 ? 0 : bays - 1 ) : Math.floor( random() * bays );

		for ( let f = 0; f < p.floors; f ++ ) {

			for ( let b = 0; b < bays; b ++ ) {

				const cx = - w / 2 + margin + ( b + 0.5 ) * pitch;

				if ( f === 0 && b === doorBay && ! back ) {

					openings.push( { kind: 'door', cx, sill: 0, ow: Math.max( 1.05, ow ), oh: 2.25 } );
					continue;

				}

				// the back is plainer and the odd bay is blank on any floor — the
				// irregularity that keeps a row from reading as copies
				const blank = back ? 0.35 : ( f === 0 ? 0.35 : 0.1 );
				if ( random() < blank ) continue;

				// the room this opening looks into: one interior per floor, spanning
				// the facade, so neighbouring windows share it
				const floorBase = f * p.floorHeight;
				const roomWidth = Math.max( 1.4, w - 1.1 );

				if ( f > 0 && ! back && random() < p.balconyChance ) {

					openings.push( { kind: 'french', cx, sill: floorBase + 0.02, ow, oh: 2.15, floorBase, roomWidth } );

				} else {

					const sill = floorBase + ( f === 0 ? 1.05 : 0.95 );
					openings.push( { kind: 'window', cx, sill, ow: back ? ow * 0.85 : ow, oh: back ? oh * 0.9 : oh, floorBase, roomWidth } );

				}

			}

		}

	}

	for ( const o of openings ) {

		const hole = new Path();
		hole.moveTo( o.cx - o.ow / 2, o.sill );
		hole.lineTo( o.cx - o.ow / 2, o.sill + o.oh );
		hole.lineTo( o.cx + o.ow / 2, o.sill + o.oh );
		hole.lineTo( o.cx + o.ow / 2, o.sill );
		hole.lineTo( o.cx - o.ow / 2, o.sill );
		shape.holes.push( hole );

	}

	const wall = new ExtrudeGeometry( shape, { depth: t, bevelEnabled: false } );
	wall.translate( 0, 0, d / 2 - t );
	if ( rotationY !== 0 ) wall.rotateY( rotationY );
	extras.push( { geometry: wall, partId: WALL } );

	// dress every opening; module positions rotate with the facade

	for ( const o of openings ) {

		addOpening( modules, o, d, p, random, rotationY );

	}

}

// the profile of a side wall, extruded to the wall thickness: a pentagon rising
// to the ridge where it carries a gable, flat-topped under a terrace; an
// exposed ( row-end ) side takes one window per upper floor
function buildGableWall( extras, modules, w, d, t, eaves, ridge, p, random, side, exposed, gabled ) {

	const halfD = d / 2 - 0.02;

	const shape = new Shape();
	shape.moveTo( - halfD, 0 );
	shape.lineTo( halfD, 0 );
	shape.lineTo( halfD, eaves );
	if ( gabled ) shape.lineTo( 0, ridge );
	shape.lineTo( - halfD, eaves );
	shape.lineTo( - halfD, 0 );

	const openings = [];

	if ( exposed ) {

		for ( let f = 1; f < p.floors; f ++ ) {

			if ( random() < 0.35 ) continue;

			const cz = ( random() - 0.5 ) * ( d * 0.3 );
			openings.push( { kind: 'window', cx: cz, sill: f * p.floorHeight + 0.95, ow: p.openingWidth * 0.85, oh: p.openingHeight * 0.9, floorBase: f * p.floorHeight, roomWidth: Math.max( 1.4, d * 0.55 ) } );

		}

	}

	for ( const o of openings ) {

		const hole = new Path();
		hole.moveTo( o.cx - o.ow / 2, o.sill );
		hole.lineTo( o.cx - o.ow / 2, o.sill + o.oh );
		hole.lineTo( o.cx + o.ow / 2, o.sill + o.oh );
		hole.lineTo( o.cx + o.ow / 2, o.sill );
		hole.lineTo( o.cx - o.ow / 2, o.sill );
		shape.holes.push( hole );

	}

	// author the profile across Z: rotate the extrusion so its depth runs along X,
	// with the outer face landing on this side's party-wall plane
	const rotationY = side < 0 ? - Math.PI / 2 : Math.PI / 2;

	const wall = new ExtrudeGeometry( shape, { depth: t, bevelEnabled: false } );
	wall.translate( 0, 0, w / 2 - t - 0.02 );
	wall.rotateY( rotationY );
	extras.push( { geometry: wall, partId: WALL } );

	for ( const o of openings ) {

		addOpening( modules, o, w - 0.04, p, random, rotationY );

	}

}

// --- opening modules -------------------------------------------------------

// dresses one punched opening: painted surround, sill, recessed glazing, louvered
// shutters — and, for a French door, the little iron balcony in front of it.
// authored on the +Z facade of a house `depth` deep, then spun into place.
function addOpening( modules, o, depth, p, random, rotationY ) {

	const { trim, shutters, rails, stone, glass, glassRooms, doors } = modules;

	const face = depth / 2; // the outer wall plane
	const cy = o.sill + o.oh / 2;

	const rotate = ( matrix ) => rotationY === 0 ? matrix : matrix.premultiply( _matrix.makeRotationY( rotationY ) );

	// painted surround: a flat band just proud of the plaster, framing the opening

	const s = SURROUND;
	const proud = 0.025;

	trim.push( rotate( boxMatrix( o.cx, o.sill + o.oh + s / 2, face + proud / 2, o.ow + s * 2, s, proud ) ) );
	trim.push( rotate( boxMatrix( o.cx - o.ow / 2 - s / 2, cy, face + proud / 2, s, o.oh + s * 2, proud ) ) );
	trim.push( rotate( boxMatrix( o.cx + o.ow / 2 + s / 2, cy, face + proud / 2, s, o.oh + s * 2, proud ) ) );

	if ( o.kind === 'door' ) {

		// door leaf set deep in the reveal, with a short flight of threshold
		// steps descending until they meet the ground, so the door stays
		// reachable however the terrace falls away in front

		doors.push( rotate( planeMatrix( o.cx, cy, face - 0.16, o.ow - 0.02, o.oh - 0.02 ) ) );

		for ( let i = 0; i < 3; i ++ ) {

			stone.push( rotate( boxMatrix( o.cx, - 0.08 - i * 0.16, face + 0.12 + i * 0.14, o.ow + 0.3 + i * 0.14, 0.16, 0.36 + i * 0.14 ) ) );

		}

		return;

	}

	// glazing, set back into the reveal so the opening keeps real depth, each
	// pane recording the floor-wide room it looks into ( in house-local space,
	// spun with the facade like everything else )

	glass.push( rotate( planeMatrix( o.cx, cy, face - 0.13, o.ow - 0.04, o.oh - 0.04 ) ) );

	const center = new Vector3( 0, o.floorBase + p.floorHeight / 2, face - 0.13 );
	if ( rotationY !== 0 ) center.applyMatrix4( _matrix.makeRotationY( rotationY ) );
	glassRooms.push( { center, size: new Vector2( o.roomWidth, p.floorHeight - 0.5 ) } );

	// sill: a small slab under the opening, proud of the surround

	trim.push( rotate( boxMatrix( o.cx, o.sill - 0.03, face + 0.045, o.ow + 0.2, 0.07, 0.14 ) ) );

	// shutters: louvered bifold panels, most folded open against the wall, the
	// rest closed into the reveal — the mix that gives a lived-in facade

	const pw = o.ow * 0.28; // one leaf of a bifold pair
	const ph = o.oh + 0.08;
	const pt = 0.035;

	if ( random() < p.closedChance ) {

		shutters.push( rotate( boxMatrix( o.cx - o.ow / 4, cy, face - 0.055, o.ow / 2 + 0.02, ph - 0.1, pt ) ) );
		shutters.push( rotate( boxMatrix( o.cx + o.ow / 4, cy, face - 0.055, o.ow / 2 + 0.02, ph - 0.1, pt ) ) );

	} else {

		const off = o.ow / 2 + s + 0.03 + pw / 2;

		shutters.push( rotate( boxMatrix( o.cx - off, cy, face + proud + pt / 2, pw, ph, pt ) ) );
		shutters.push( rotate( boxMatrix( o.cx + off, cy, face + proud + pt / 2, pw, ph, pt ) ) );

	}

	if ( o.kind === 'french' ) {

		// the balcony: a thin slab and an iron rail of round-ish bars — the
		// balusters are boxes, kept few and thin so the rail stays light

		const bw = o.ow + 0.7;
		const bd = 0.7;
		const slabY = o.sill - 0.045;

		trim.push( rotate( boxMatrix( o.cx, slabY, face + bd / 2, bw, 0.09, bd ) ) );

		const railY = slabY + 0.5;
		const railTop = slabY + 0.95;

		rails.push( rotate( boxMatrix( o.cx, railTop, face + bd - 0.03, bw, 0.04, 0.04 ) ) );
		rails.push( rotate( boxMatrix( o.cx - bw / 2 + 0.03, railTop, face + bd / 2, 0.04, 0.04, bd - 0.06 ) ) );
		rails.push( rotate( boxMatrix( o.cx + bw / 2 - 0.03, railTop, face + bd / 2, 0.04, 0.04, bd - 0.06 ) ) );

		const count = Math.max( 3, Math.round( bw / 0.16 ) );

		for ( let i = 0; i <= count; i ++ ) {

			const bx = o.cx - bw / 2 + 0.03 + ( bw - 0.06 ) * i / count;
			rails.push( rotate( boxMatrix( bx, railY, face + bd - 0.03, 0.022, 0.9, 0.022 ) ) );

		}

	}

}

// a unit plane ( facing +Z ) scaled and placed on the front facade; the caller's
// rotate() spins it onto the other faces
function planeMatrix( x, y, z, sizeX, sizeY ) {

	return new Matrix4()
		.makeScale( sizeX, sizeY, 1 )
		.setPosition( _position.set( x, y, z ) );

}

// --- roof ------------------------------------------------------------------

// tiled slopes over the gabled span: two low-pitched planes with overhanging
// eaves, verge tiles, a ridge cap, fascia boards and a cornice line on each
// facade — and, where the span stops short of the house end, a gable-end wall
// closing the roof against the terrace
function buildRoof( extras, trim, brick, w, d, eaves, ridge, p, span ) {

	const [ x0, x1 ] = span;

	// overhang past the house ends only; at an interior split the roof stays
	// flush so the terrace parapet can meet it
	const atLeft = x0 < - w / 2 + 0.01;
	const atRight = x1 > w / 2 - 0.01;
	const overhangL = atLeft ? 0.18 : 0.04;
	const overhangR = atRight ? 0.18 : 0.04;
	const overhangEave = 0.4;

	const width = ( x1 - x0 ) + overhangL + overhangR;
	const cx = ( x0 + x1 ) / 2 + ( overhangR - overhangL ) / 2;

	const run = d / 2 + overhangEave;
	const eaveY = eaves - overhangEave * p.roofPitch + 0.12; // the roof plane floats a beam's depth above the wall head
	const ridgeY = ridge + 0.12;
	const slopeLength = Math.hypot( run, ridgeY - eaveY );

	const angle = Math.atan2( run, ridgeY - eaveY );
	const tilt = Math.atan2( ridgeY - eaveY, run ); // the slope's pitch off the horizontal

	for ( const side of [ 1, - 1 ] ) {

		const slope = metricUV( new PlaneGeometry( width, slopeLength ), width, slopeLength );
		slope.rotateX( - angle );
		slope.translate( 0, ( ridgeY + eaveY ) / 2, run / 2 );
		if ( side < 0 ) slope.rotateY( Math.PI );
		slope.translate( cx, 0, 0 );
		extras.push( { geometry: slope, partId: ROOF } );

		// verge tiles along the gable edges, so the thin slope plane reads as a
		// roof with body from street level rather than vanishing edge-on
		const lean = side > 0 ? tilt : Math.PI - tilt;

		for ( const sx of [ - 1, 1 ] ) {

			brick.push( new Matrix4()
				.makeRotationX( lean )
				.scale( _scale.set( 0.14, 0.12, slopeLength - 0.02 ) )
				.setPosition( _position.set( cx + sx * ( width / 2 - 0.06 ), ( ridgeY + eaveY ) / 2 + 0.02, side * run / 2 ) ) );

		}

		// fascia board tucked under the eave edge, and the cornice line where
		// the wall head meets the roof — the crisp white eaves of the coast

		trim.push( boxMatrix( cx, eaveY - 0.09, side * ( d / 2 + overhangEave - 0.03 ), width, 0.15, 0.06 ) );
		trim.push( boxMatrix( cx, eaves - 0.06, side * ( d / 2 + 0.05 ), x1 - x0 - 0.06, 0.13, 0.1 ) );

	}

	brick.push( boxMatrix( cx, ridgeY + 0.03, 0, width, 0.1, 0.34 ) );

	// the interior gable end: a triangle standing on the wall head, closing the
	// slopes where they stop over the terrace; tucked just inside the gabled span
	for ( const [ at, x, inset ] of [[ atLeft, x0, 0 ], [ atRight, x1, - 0.22 ]] ) {

		if ( at ) continue;

		const halfD = d / 2 - 0.02;
		const shape = new Shape();
		shape.moveTo( - halfD, eaves - 0.05 );
		shape.lineTo( halfD, eaves - 0.05 );
		shape.lineTo( 0, ridge );
		shape.lineTo( - halfD, eaves - 0.05 );

		const wall = new ExtrudeGeometry( shape, { depth: 0.22, bevelEnabled: false } );
		wall.translate( 0, 0, x + inset );
		wall.rotateY( Math.PI / 2 );
		extras.push( { geometry: wall, partId: WALL } );

	}

}

// a walkable roof terrace over the flat span: a cotto tile floor behind a
// plastered parapet — with a light iron rail on the balcony half of a combo
// roof, where the terrace shares the top with a gable
function buildTerrace( wallBoxes, trim, rails, terraceBoxes, w, d, eaves, span, balcony ) {

	const [ x0, x1 ] = span;
	const width = x1 - x0;
	const cx = ( x0 + x1 ) / 2;

	terraceBoxes.push( boxMatrix( cx, eaves - 0.02, 0, width - 0.04, 0.12, d - 0.1 ) );

	const ph = balcony ? 0.5 : 0.85;

	const parapets = [
		[ cx, d / 2 - 0.09, width - 0.02, 0.16 ],
		[ cx, - d / 2 + 0.09, width - 0.02, 0.16 ]
	];

	if ( x0 < - w / 2 + 0.01 ) parapets.push( [ x0 + 0.09, 0, 0.16, d - 0.36 ] );
	if ( x1 > w / 2 - 0.01 ) parapets.push( [ x1 - 0.09, 0, 0.16, d - 0.36 ] );

	for ( const [ px, pz, sx, sz ] of parapets ) {

		wallBoxes.push( boxMatrix( px, eaves + ph / 2, pz, sx, ph, sz ) );
		trim.push( boxMatrix( px, eaves + ph + 0.03, pz, sx + 0.06, 0.07, sz + 0.06 ) );

	}

	if ( balcony ) {

		// the rail over the front parapet, so the terrace reads as the balcony
		rails.push( boxMatrix( cx, eaves + ph + 0.45, d / 2 - 0.09, width - 0.08, 0.035, 0.035 ) );

		const count = Math.max( 3, Math.round( width / 0.5 ) );

		for ( let i = 0; i <= count; i ++ ) {

			rails.push( boxMatrix( x0 + 0.06 + ( width - 0.12 ) * i / count, eaves + ph + 0.24, d / 2 - 0.09, 0.025, 0.42, 0.025 ) );

		}

	}

}

// --- material ----------------------------------------------------------------

// derivative-based bump for a procedural, world-space height field. the built-in bumpMap
// offsets the UV to read its height, so it returns a zero gradient for a height keyed off
// world position; this feeds the hardware screen-space derivatives of the height into
// Mikkelsen's surface-gradient method so the relief actually perturbs the normal.
function bumpNormal( height ) {

	const dpdx = positionView.dFdx();
	const dpdy = positionView.dFdy();
	const r1 = dpdy.cross( normalView );
	const r2 = normalView.cross( dpdx );
	const det = dpdx.dot( r1 );
	const grad = det.sign().mul( height.dFdx().mul( r1 ).add( height.dFdy().mul( r2 ) ) );

	return det.abs().mul( normalView ).sub( grad ).normalize();

}

// cheap value noise ( ~[ -1, 1 ] ), integer-hashed ( not fract(sin) ) to stay
// stable across drivers, and far lighter than gradient mx_noise per tap
const valueNoise = /*@__PURE__*/ Fn( ( [ p ] ) => {

	const i = floor( p );
	const f = fract( p );
	const u = f.mul( f ).mul( f.mul( - 2 ).add( 3 ) ); // 3f2 - 2f3 smooth interpolation
	const corner = ( ox, oy, oz ) => {

		const c = i.add( vec3( ox, oy, oz ) );
		return ihash( uint( c.x.add( 1 << 20 ) ).mul( uint( 73856093 ) ).bitXor( uint( c.y.add( 1 << 20 ) ).mul( uint( 19349663 ) ) ).bitXor( uint( c.z.add( 1 << 20 ) ).mul( uint( 83492791 ) ) ) );

	};

	const x00 = mix( corner( 0, 0, 0 ), corner( 1, 0, 0 ), u.x );
	const x10 = mix( corner( 0, 1, 0 ), corner( 1, 1, 0 ), u.x );
	const x01 = mix( corner( 0, 0, 1 ), corner( 1, 0, 1 ), u.x );
	const x11 = mix( corner( 0, 1, 1 ), corner( 1, 1, 1 ), u.x );
	return mix( mix( x00, x10, u.y ), mix( x01, x11, u.y ), u.z ).mul( 2 ).sub( 1 );

} ).setLayout( { name: 'valueNoise', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

// antialiased repeated line at every multiple of `period` ( tile joints and
// courses ): the drawn line never falls below the pixel footprint, so the
// pattern stays legible into the distance and dissolves instead of shimmering
function gridLine( coord, period, halfWidth ) {

	const g = coord.div( period );
	const d = float( 0.5 ).sub( fract( g ).sub( 0.5 ).abs() ); // distance to nearest line, in periods
	const aa = fwidth( g ).max( 0.0001 );
	const hw = halfWidth / period;
	return smoothstep( float( hw ).add( aa ), float( hw ).sub( aa ), d ).mul( float( hw ).div( fwidth( g ).max( hw ) ).min( 1 ) );

}

// fractal ( fBm ) of valueNoise; unrolled for a compile-time octave count
const valueFractal = ( p, octaves ) => {

	let sum = valueNoise( p );
	let amp = 0.5, freq = 2;
	for ( let o = 1; o < octaves; o ++ ) {

		sum = sum.add( valueNoise( p.mul( freq ) ).mul( amp ) );
		amp *= 0.5;
		freq *= 2;

	}

	return sum;

};

// interior mapping: fakes a furnished room behind each pane in the fragment
// shader — no geometry, no texture. every pane carries the room it looks into
// ( centre + size, baked per opening, one room per floor ), so the windows of a
// floor share one interior. the view ray is cast into that box and whatever it
// meets — cotto floor, beamed ceiling, whitewashed walls, a few pieces of dark
// old furniture — is shaded procedurally from a per-room hash. returns
// vec4( colour, lit ).
const interior = /*@__PURE__*/ Fn( () => {

	// flat so floor() below can't split one pane across two cell ids ( centre is per-room )
	const roomCenter = varying( attribute( 'roomCenter', 'vec3' ) ).setInterpolation( InterpolationSamplingType.FLAT, InterpolationSamplingMode.EITHER );
	const roomSize = attribute( 'roomSize', 'vec2' ).max( vec2( 1.3, 1.5 ) ); // roomless glass ( the campanile's belfry ) still resolves a plausible dark chamber

	// a per-face frame from the geometry normal ( holds on every facade ):
	// u runs across the face, v is up, n points outward
	const n = normalLocal;
	const up = vec3( 0, 1, 0 );
	const uAxis = cross( up, n ).normalize();

	// this pixel and the view ray, in the room's ( across, up, depth ) frame;
	// depth runs into the wall, so the ray's depth component is positive
	const d = positionLocal.sub( roomCenter );
	const camLocal = modelWorldMatrixInverse.mul( vec4( cameraPosition, 1 ) ).xyz;
	const rayLocal = positionLocal.sub( camLocal ).normalize();
	const origin = vec3( dot( d, uAxis ), d.y, 0 );
	const dir = vec3( dot( rayLocal, uAxis ), rayLocal.y, dot( rayLocal, n ).negate() );

	// the room box: floor-wide and floor-high, set back behind the glass and a
	// little deeper than it is tall. shade the far side the ray exits ( slab
	// method: nearest of the three far-plane crossings; dividing by a near-zero
	// direction gives ±inf, which min() harmlessly drops ).
	const setback = float( 0.08 );
	const boxMax = vec3( roomSize.x.mul( 0.5 ), roomSize.y.mul( 0.5 ), setback.add( roomSize.y.mul( 1.6 ) ) );
	const boxMin = vec3( boxMax.x.negate(), boxMax.y.negate(), setback );
	const tFar = boxMin.sub( origin ).div( dir ).max( boxMax.sub( origin ).div( dir ) );
	const t = tFar.x.min( tFar.y ).min( tFar.z );
	const hit = origin.add( dir.mul( t ) );
	const q = hit.sub( boxMin ).div( boxMax.sub( boxMin ) ); // 0..1 inside the room

	const onBack = q.z.greaterThan( 0.998 );
	const onCeil = q.y.greaterThan( 0.998 );
	const onFloor = q.y.lessThan( 0.002 );

	// per-room key for a portable integer hash — fract( sin() ) isn't bit-exact
	// across drivers. rooms bake in house-local space, where different houses'
	// floors land on near-identical centres, so the per-house id is folded in
	// to keep every interior its own.
	const cell = floor( roomCenter.mul( 2.0 ) );
	const ckey = uint( cell.x.add( 1 << 21 ) ).mul( uint( 73856093 ) )
		.bitXor( uint( cell.y.add( 1 << 21 ) ).mul( uint( 19349663 ) ) )
		.bitXor( uint( cell.z.add( 1 << 21 ) ).mul( uint( 83492791 ) ) )
		.bitXor( uint( attribute( 'houseId', 'float' ).mul( 65535 ) ) ).toVar();
	const hash = ( kx, ky, kz ) => ihash( ckey.add( uint( Math.round( ( kx + ky * 7 + kz * 13 ) * 100 ) ) ) );
	const seed = hash( 12.9898, 78.233, 37.719 );
	const seed2 = hash( 39.346, 11.135, 83.155 );
	const lit = step( 0.84, hash( 63.21, 9.17, 51.43 ) ); // a few lamps burn even in daylight; toward dusk they carry the facade
	const lightCol = mix( color( 0xffb845 ), color( 0xffdf9a ), hash( 27.1, 4.9, 61.7 ) ); // village bulbs all run warm, candle-glow to tungsten

	// depth falloff ( dimmer toward the back, kept warm — daylight bounces off
	// these pale walls ), and a panel mask on a face given its two 0..1
	// coordinates — used for the flat fittings below
	const depth = roomSize.y.mul( 1.6 );
	const falloffAt = ( z ) => mix( vec3( 1.0, 1.0, 1.0 ), vec3( 0.52, 0.47, 0.4 ), z.sub( setback ).div( depth ).clamp( 0, 1 ) );
	const rect = ( ax, ay, cx, cy, hw, hh ) => smoothstep( hw + 0.006, hw - 0.006, ax.sub( cx ).abs() ).mul( smoothstep( hh + 0.006, hh - 0.006, ay.sub( cy ).abs() ) );

	// --- the room shell: walls, floor, ceiling, back wall ---------------------

	// whitewash and warm plasters — an ochre wash in some rooms, a muted sage in
	// a few parlours — over a darker skirting line at the wall foot
	let wall = mix( color( 0xcfc4ac ), color( 0xdcd1b8 ), seed );
	wall = select( hash( 5.5, 2.2, 8.8 ).greaterThan( 0.72 ), mix( color( 0xbda374 ), color( 0xcbb384 ), seed ), wall );
	wall = select( hash( 9.1, 3.3, 1.2 ).greaterThan( 0.92 ), mix( color( 0x97a48c ), color( 0xa7b29a ), seed ), wall );
	const wallCol = mix( wall, wall.mul( 0.45 ), smoothstep( 0.06, 0.05, q.y ) );

	// a cotto tile floor — the same fired orange as the roofs — under a woven rug
	const tileSeam = step( 0.92, fract( q.x.mul( 7 ) ) ).max( step( 0.92, fract( q.z.mul( 9 ) ) ) );
	const cotto = mix( color( 0x8a4f30 ), color( 0xa8663f ), seed ).mul( tileSeam.mul( 0.3 ).oneMinus() );
	const rug = mix( color( 0x7a3b32 ), color( 0x51604f ), seed2 );
	const floorCol = mix( cotto, rug, rect( q.x, q.z, 0.5, 0.6, 0.28, 0.24 ).mul( 0.9 ) );

	// the ceiling: dark chestnut beams over pale plaster, with a small lamp
	// mid-room; in a lit room the fixture reads bright and glows ( the
	// material's emissive = colour × lit )
	const beams = step( 0.84, fract( q.x.mul( mix( float( 4 ), float( 6 ), seed ) ) ) );
	const lamp = smoothstep( 0.15, 0.12, vec2( q.x.sub( 0.5 ), q.z.sub( 0.5 ) ).length() );
	let ceilCol = mix( mix( wall, color( 0xf2ecdc ), 0.55 ), color( 0x453424 ), beams.mul( 0.85 ) );
	ceilCol = mix( ceilCol, lightCol.mul( mix( float( 1.0 ), float( 4.5 ), lit ) ), lamp );

	// back wall: a panelled door to one side, and a small dark-framed print
	// kept on the opposite half of the wall so it never lands on the door
	const doorX = mix( float( 0.22 ), float( 0.78 ), seed );
	const door = mix( color( 0x5a4631 ), color( 0x6b5136 ), seed2 );
	const picX = select( doorX.lessThan( 0.5 ), mix( float( 0.66 ), float( 0.84 ), seed2 ), mix( float( 0.16 ), float( 0.34 ), seed2 ) );
	const picCol = mix( color( 0x39506b ), color( 0x8a5a3a ), hash( 5.1, 9.2, 3.3 ) );
	let backCol = mix( wallCol, door, rect( q.x, q.y, doorX, 0.34, 0.085, 0.36 ) );
	backCol = mix( backCol, color( 0x1c1712 ), rect( q.x, q.y, picX, 0.58, 0.06, 0.075 ) ); // dark frame
	backCol = mix( backCol, picCol, rect( q.x, q.y, picX, 0.58, 0.045, 0.06 ) ); // the print

	const shellCol = select( onBack, backCol, select( onCeil, ceilCol, select( onFloor, floorCol, wallCol ) ) );

	// fake ambient occlusion: darken the hit toward the room's edges ( where two
	// surfaces meet ), so the box reads with soft corner shading instead of
	// flat-lit walls
	const aoBand = 0.15;
	const aoEdge = ( a ) => smoothstep( 0, aoBand, a ).mul( smoothstep( 0, aoBand, a.oneMinus() ) );
	const edgeAO = select( onBack, aoEdge( q.x ).mul( aoEdge( q.y ) ), select( onFloor.or( onCeil ), aoEdge( q.x ).mul( aoEdge( q.z ) ), aoEdge( q.y ).mul( aoEdge( q.z ) ) ) );
	const shellAO = mix( float( 0.7 ), float( 1.0 ), edgeAO );

	// --- nearest surface: the shell, then any furniture that stands closer ----
	// each piece is a solid axis-aligned box in room space; boxHit returns its
	// near face. consider() keeps whichever surface the ray meets first.
	let bestT = t;
	let bestCol = shellCol.mul( shellAO ).mul( falloffAt( hit.z ) );
	let bestEmit = float( 1 ); // per-hit emissive weight: shell and fittings emit fully, sheers far less

	const boxHit = ( bMin, bMax ) => {

		const ta = bMin.sub( origin ).div( dir );
		const tb = bMax.sub( origin ).div( dir );
		const lo = ta.min( tb ), hi = ta.max( tb );
		const tN = lo.x.max( lo.y ).max( lo.z );
		const p = origin.add( dir.mul( tN ) );
		return { tN, p, hit: hi.x.min( hi.y ).min( hi.z ).greaterThan( tN ).and( tN.greaterThan( 0 ) ), qb: p.sub( bMin ).div( bMax.sub( bMin ) ) };

	};

	const consider = ( h, tN, c, emit = 1 ) => {

		const near = h.and( tN.lessThan( bestT ) ); bestCol = select( near, c, bestCol ); bestEmit = select( near, float( emit ), bestEmit ); bestT = select( near, tN, bestT );

	};

	const halfU = boxMax.x, floorY = boxMin.y, ceilY = boxMax.y, backZ = boxMax.z;
	const midZ = setback.add( depth.mul( 0.5 ) );

	// the kitchen table, mid-room, in waxed olive wood ( its top catches the light )
	const tCx = mix( float( - 0.5 ), float( 0.5 ), seed );
	const tCz = midZ.add( mix( float( - 0.4 ), float( 0.4 ), seed2 ) );
	const tbl = boxHit( vec3( tCx.sub( 0.55 ), floorY, tCz.sub( 0.35 ) ), vec3( tCx.add( 0.55 ), floorY.add( 0.45 ), tCz.add( 0.35 ) ) );
	const tblCol = mix( color( 0x6b4a28 ), color( 0x8a6335 ), seed2 ).mul( select( tbl.qb.y.greaterThan( 0.93 ), float( 1.25 ), float( 0.8 ) ) );
	consider( tbl.hit, tbl.tN, tblCol.mul( falloffAt( tbl.p.z ) ) );

	// a low credenza against the back wall, its top set with a pale runner
	const crCx = mix( halfU.mul( - 0.3 ), halfU.mul( 0.3 ), seed2 );
	const cr = boxHit( vec3( crCx.sub( 0.8 ), floorY, backZ.sub( 0.5 ) ), vec3( crCx.add( 0.8 ), floorY.add( mix( float( 0.85 ), float( 1.0 ), seed ) ), backZ.sub( 0.08 ) ) );
	const crCol = mix( color( 0x453222 ), color( 0x5c452e ), seed ).mul( select( cr.qb.y.greaterThan( 0.92 ), float( 1.3 ), float( 0.82 ) ) );
	consider( cr.hit, cr.tN, crCol.mul( falloffAt( cr.p.z ) ) );

	// tall madie ( dressers ) in the back corners — each side stands in some rooms
	const dresser = ( cx, gate, h ) => {

		const w = boxHit( vec3( cx.sub( 0.45 ), floorY, backZ.sub( 0.55 ) ), vec3( cx.add( 0.45 ), floorY.add( h ), backZ.sub( 0.08 ) ) );
		const c = mix( color( 0x33261a ), color( 0x4a392a ), seed ).mul( select( w.qb.y.greaterThan( 0.93 ), float( 1.2 ), float( 0.82 ) ) );
		consider( w.hit.and( gate ), w.tN, c.mul( falloffAt( w.p.z ) ) );

	};

	dresser( halfU.mul( - 0.82 ), hash( 7.3, 2.1, 9.9 ).greaterThan( 0.45 ), mix( float( 1.6 ), float( 2.1 ), seed ) );
	dresser( halfU.mul( 0.82 ), hash( 3.7, 8.4, 1.5 ).greaterThan( 0.45 ), mix( float( 1.6 ), float( 2.1 ), seed2 ) );

	// sheers hung just inside the glass: lace and ecru mostly, the odd striped
	// or terracotta pair, drawn part-way in from each side
	const swatch = ( a, b ) => mix( color( a ), color( b ), seed2 );
	const pick = hash( 22.4, 6.7, 91.2 ).mul( 4 ); // 0..4, one bucket per family
	let fabric = swatch( 0xe0dacb, 0xece7d9 ); // white lace — the default on every alley
	fabric = select( pick.greaterThan( 1.6 ), swatch( 0xcfc4a8, 0xdcd2b8 ), fabric ); // ecru
	fabric = select( pick.greaterThan( 2.8 ), swatch( 0x9a8f72, 0xb0a586 ), fabric ); // faded olive stripe
	fabric = select( pick.greaterThan( 3.6 ), swatch( 0x9a6a52, 0xac7a5e ), fabric ); // terracotta
	const drape = ( bMin, bMax, gate ) => {

		const h = boxHit( bMin, bMax );
		const pleat = fabric.mul( mix( float( 0.8 ), float( 1.1 ), fract( h.p.x.mul( 3.5 ) ) ) ); // soft vertical pleats
		consider( h.hit.and( gate ), h.tN, pleat.mul( falloffAt( h.p.z ) ), 0.3 ); // a sheer only mutes the room's glow, never blocks it

	};

	const cz0 = setback, cz1 = setback.add( 0.1 );
	// sheer widths, biased narrow ( squared ) and each capped at half the room
	// width, so most windows read airy and open
	const sL = smoothstep( 0.3, 1.0, seed ), sR = smoothstep( 0.3, 1.0, seed2 );
	const lw = halfU.mul( sL.mul( sL ) );
	const rw = halfU.mul( sR.mul( sR ) );
	drape( vec3( halfU.negate(), floorY, cz0 ), vec3( halfU.negate().add( lw ), ceilY, cz1 ), lw.greaterThan( 0.05 ) );
	drape( vec3( halfU.sub( rw ), floorY, cz0 ), vec3( halfU, ceilY, cz1 ), rw.greaterThan( 0.05 ) );

	// lit rooms read brighter and take on the warmth of their bulb
	const warmth = mix( vec3( 1.0, 1.0, 1.0 ), lightCol, lit.mul( 0.85 ) );
	return vec4( bestCol.mul( warmth ).mul( mix( float( 1.0 ), float( 1.35 ), lit ) ), lit.mul( bestEmit ) );

} );

/**
 * The colour-washed palette of the Ligurian coast ( hex colours ): ochres and
 * warm yellows lead, cut with terracotta oranges, the odd deep coral red, dusty
 * pinks and cream. Common tones repeat so an equal-probability pick lands on
 * them more often, the way the real villages read.
 */
const housePalette = [
	0xc2503a, 0xb04a38, // deep coral red ( the accent every postcard finds )
	0xc96f42, 0xd07a4a, // burnt orange / terracotta
	0xd88f3e, 0xcd8a45, // ochre orange
	0xd9a84e, 0xdcae57, 0xd2a24b, // golden ochre — the workhorse
	0xdec06f, 0xe3c87e, // pale straw yellow
	0xcf9a86, 0xd4a494, // dusty pink
	0xe4d7b6, 0xece2c8, // warm cream / whitewash
	0xb9835f // faded sienna
];

/**
 * The single material every house ( and the village's stonework ) is dressed
 * with: it reads the baked per-vertex `partId` and reproduces every zone —
 * weathered colour-washed plaster on the walls, painted surrounds and cornices,
 * louvered shutters, barrel-tiled terracotta roofs, dark recessed glazing, plank
 * doors, rubble-stone podiums and iron rails. The wall colour comes from the
 * baked `paint` palette index, and `houseId` drives per-house drift, so one
 * material covers the whole village in a handful of shader branches.
 */
function createHouseMaterial() {

	// the palette pick is constant across a house, so resolve it per-vertex
	const palette = housePalette.map( hex => color( hex ) );
	const paint = attribute( 'paint', 'float' );

	let paintColor = palette[ 0 ];
	for ( let i = 1; i < palette.length; i ++ ) paintColor = mix( paintColor, palette[ i ], step( i - 0.5, paint ) );

	// per-house hash lane: houseId is a baked 0..1 constant, quantized to a key
	const houseKey = uint( attribute( 'houseId', 'float' ).mul( 65535 ) ).toVar();
	const houseHash = ( k ) => ihash( houseKey.add( uint( k ) ) );

	const wallBase = varying( paintColor.mul( houseHash( 17 ).mul( 0.16 ).add( 0.92 ) ) ); // subtle per-house brightness

	// broad weathering, world-keyed so it reads consistently across the village:
	// a slow tonal drift, a fine plaster mottle, and streaks bleeding down walls

	const tone = varying( valueNoise( positionWorld.mul( 0.05 ) ) ).mul( 0.1 );
	const mottle = valueNoise( positionWorld.mul( 1.6 ) ).mul( 0.045 );
	const streak = valueFractal( vec3( positionWorld.x.mul( 1.9 ), positionWorld.y.mul( 0.06 ), positionWorld.z.mul( 1.9 ) ), 2 );
	const grime = smoothstep( 0.05, 0.6, streak ).mul( 0.14 );

	// rising damp and plaster scabbed off to stone near the wall base — houses bake
	// in local space with the ground floor at y = 0, so local height keys both
	const baseness = smoothstep( 1.8, 0.1, positionLocal.y );
	const scab = smoothstep( 0.55, 0.8, valueFractal( positionWorld.mul( 0.55 ), 2 ).mul( 0.5 ).add( 0.5 ) ).mul( baseness );

	const stoneTone = mix( color( 0x8f8674 ), color( 0xa39a86 ), valueNoise( positionWorld.mul( 0.9 ) ).mul( 0.5 ).add( 0.5 ) );

	let plaster = wallBase.mul( float( 1 ).add( tone ).add( mottle ) );
	plaster = mix( plaster, plaster.mul( 0.72 ), grime.add( baseness.mul( 0.18 ) ) );
	plaster = mix( plaster, stoneTone, scab.mul( 0.65 ) );

	// painted trim: the warm white every surround and cornice is cut from
	const trimColor = color( 0xefe6d2 ).mul( float( 1 ).add( tone.mul( 0.5 ) ) ).mul( houseHash( 23 ).mul( 0.08 ).add( 0.95 ) );

	// shutters: per-house colour — green leads, brown and weathered teal follow
	const shutterPick = houseHash( 31 );
	const shutterDrift = houseHash( 37 );
	let shutterColor = mix( color( 0x2e4a34 ), color( 0x44634a ), shutterDrift );
	shutterColor = select( shutterPick.greaterThan( 0.72 ), mix( color( 0x54432f ), color( 0x6a5741 ), shutterDrift ), shutterColor );
	shutterColor = select( shutterPick.greaterThan( 0.9 ), mix( color( 0x5c7472 ), color( 0x718984 ), shutterDrift ), shutterColor );

	// louver relief: horizontal slats at a real ~5 cm pitch ( world Y ), faded out
	// before a slat nears a pixel so the blinds never shimmer at distance
	const texel = fwidth( positionWorld ).length(); // on-screen size of a surface pixel — our hand-rolled LOD
	const louverDetail = smoothstep( 0.045, 0.012, texel );
	const louver = fract( positionWorld.y.div( 0.05 ) ).mul( louverDetail );
	const shutterShade = shutterColor.mul( louver.mul( 0.35 ).add( 0.78 ) ).mul( float( 1 ).add( tone ) );

	// per-house roof family: terracotta for most, weathered brown or faded red
	// for the rest — and the red family skips the red-painted houses, so a roof
	// never matches its own walls
	const roofPick = houseHash( 53 );
	let roofBase = mix( color( 0x9d5231 ), color( 0xbb6c41 ), houseHash( 59 ) );
	roofBase = select( roofPick.greaterThan( 0.78 ), mix( color( 0x7c4933 ), color( 0x94593d ), houseHash( 59 ) ), roofBase );
	roofBase = select( roofPick.greaterThan( 0.92 ).and( paint.greaterThan( 1.5 ) ), mix( color( 0x8e4036 ), color( 0xa54c3e ), houseHash( 59 ) ), roofBase );

	// slope tiles: staggered terracotta barrels in metric UVs. the course and gap
	// joints are antialiased lines, so the tiling stays legible into the distance
	// instead of washing flat; only the per-tile jitter fades, so it can't sparkle.
	const course = floor( uv().y.div( 0.42 ) );
	const barrelCoord = uv().x.div( 0.19 ).add( course.mul( 0.5 ) ); // half-tile stagger per course
	const barrel = fract( barrelCoord ).sub( 0.5 ).abs().mul( 2 ).oneMinus(); // 1 on a barrel's crest, 0 in the pan
	const tileKey = uint( floor( barrelCoord ).add( 1 << 16 ) ).mul( uint( 73856093 ) ).bitXor( uint( course.add( 1 << 16 ) ).mul( uint( 19349663 ) ) );

	const tileFade = smoothstep( 0.5, 0.08, texel );
	const barrelFade = smoothstep( 0.3, 0.06, texel );

	let tile = roofBase.mul( ihash( tileKey ).sub( 0.5 ).mul( 0.26 ).mul( tileFade ).add( 1 ) );
	tile = mix( tile, color( 0x9c8a5e ), smoothstep( 0.55, 0.85, valueNoise( positionWorld.mul( 0.4 ) ) ).mul( 0.4 ) ); // lichen blooming over older slopes

	const courseJoint = gridLine( uv().y, 0.42, 0.05 );
	const barrelJoint = gridLine( barrelCoord, 1, 0.07 );
	const barrelShade = barrel.mul( 0.3 ).sub( 0.15 ).mul( barrelFade );
	const roofColor = tile
		.mul( float( 1 ).add( barrelShade ) )
		.mul( courseJoint.mul( 0.3 ).oneMinus() )
		.mul( barrelJoint.mul( 0.26 ).oneMinus() )
		.mul( float( 1 ).add( tone ) );

	// ridge caps, verge tiles and chimney stacks: solid fired clay in the roof's
	// family — coursed like stacked brick on their vertical faces, plain on top,
	// so nothing stretches with the size of the piece it dresses
	const brickCourse = gridLine( positionWorld.y, 0.075, 0.007 ).mul( smoothstep( 0.55, 0.25, normalWorldGeometry.y.abs() ) );
	const brickColor = roofBase
		.mul( valueNoise( positionWorld.mul( 3.2 ) ).mul( 0.12 ).add( 1 ) )
		.mul( brickCourse.mul( 0.24 ).oneMinus() )
		.mul( float( 1 ).add( tone ) );

	// terrace floors: square cotto tiles laid in house-local metres — always the
	// fired orange of the kiln, whatever colour the walls below are washed
	const cottoJoint = gridLine( positionLocal.x, 0.34, 0.014 ).max( gridLine( positionLocal.z, 0.34, 0.014 ) );
	const cottoCell = uint( floor( positionLocal.x.div( 0.34 ) ).add( 1 << 16 ) ).mul( uint( 73856093 ) ).bitXor( uint( floor( positionLocal.z.div( 0.34 ) ).add( 1 << 16 ) ).mul( uint( 19349663 ) ) );
	const cottoColor = mix( color( 0xa8603a ), color( 0xc07a4e ), houseHash( 71 ) )
		.mul( ihash( cottoCell ).sub( 0.5 ).mul( 0.16 ).mul( tileFade ).add( 1 ) )
		.mul( cottoJoint.mul( 0.3 ).oneMinus() )
		.mul( float( 1 ).add( tone ) );

	// glazing: the interior-mapped room is the base colour, seen through a light
	// film of dust and shade — village glass runs cleaner than a city's — and
	// the smooth surface still carries the sky's reflection over it. toVar so
	// the raymarch runs once, shared by the colour and emissive outputs.
	const room = interior().toVar();
	const pane = valueNoise( positionWorld.mul( 1.1 ) ).mul( 0.5 ).add( 0.5 );
	const shadeGlass = mix( color( 0x1c2126 ), color( 0x2e363c ), pane );
	const haze = pane.mul( 0.18 ).add( 0.32 ); // enough film that the panes read as glass, not open holes
	const glassColor = mix( room.xyz.mul( color( 0xc3ccc2 ) ), shadeGlass, haze ); // faint soda-lime tint over the room

	// doors: vertical planks in dark wood, seams keyed to the leaf's metric UV
	const plank = fract( uv().x.mul( 4.5 ) );
	const plankSeam = smoothstep( 0.0, 0.08, plank ).mul( smoothstep( 1.0, 0.92, plank ) );
	const doorColor = mix( color( 0x3d2f21 ), color( 0x5a452e ), houseHash( 41 ) ).mul( plankSeam.mul( 0.25 ).add( 0.75 ) );

	// stonework: rubble masonry — coarse block mottle over rough coursing, washed
	// darker toward the waterline with a band of algae at the swell
	const block = valueNoise( vec3( positionWorld.x.mul( 1.3 ), positionWorld.y.mul( 2.6 ), positionWorld.z.mul( 1.3 ) ) ).mul( 0.5 ).add( 0.5 );
	let stoneColor = mix( color( 0x776f5f ), color( 0x958b76 ), block );
	stoneColor = stoneColor.mul( valueNoise( positionWorld.mul( 0.35 ) ).mul( 0.14 ).add( 0.93 ) );
	stoneColor = stoneColor.mul( valueNoise( positionWorld.mul( 5.5 ) ).mul( 0.1 ).mul( smoothstep( 0.06, 0.015, texel ) ).add( 1 ) ); // close-range grit, faded before it can sparkle
	stoneColor = mix( stoneColor, color( 0x4c5348 ), smoothstep( 1.6, 0.5, positionWorld.y ).mul( 0.5 ) ); // algae and tide-wash above the sea
	stoneColor = mix( stoneColor, color( 0x3d4640 ), smoothstep( 0.7, 0.1, positionWorld.y ).mul( 0.5 ) );

	const railColor = color( 0x211f1c );

	// the merged geometry carries a per-vertex partId; this material reads it and
	// branches to reproduce each zone — no per-part materials, one draw per house

	const partId = varying( attribute( 'partId', 'float' ) ).setInterpolation( InterpolationSamplingType.FLAT, InterpolationSamplingMode.EITHER ); // flat: a per-face id must not interpolate, or equal() below misses on the rounding
	const isTrim = partId.equal( TRIM );
	const isShutter = partId.equal( SHUTTER );
	const isRoof = partId.equal( ROOF );
	const isGlass = partId.equal( GLASS );
	const isDoor = partId.equal( DOOR );
	const isStone = partId.equal( STONE );
	const isRail = partId.equal( RAIL );
	const isTerrace = partId.equal( TERRACE );
	const isBrick = partId.equal( BRICK );

	const material = new MeshStandardNodeMaterial();

	material.colorNode = select( isTrim, trimColor,
		select( isShutter, shutterShade,
			select( isRoof, roofColor,
				select( isBrick, brickColor,
					select( isTerrace, cottoColor,
						select( isGlass, glassColor,
							select( isDoor, doorColor,
								select( isStone, stoneColor,
									select( isRail, railColor, plaster ) ) ) ) ) ) ) ) );

	material.roughnessNode = select( isGlass, float( 0.14 ), select( isRail, float( 0.5 ), select( isTerrace, float( 0.85 ), float( 0.95 ) ) ) );
	material.metalnessNode = select( isRail, float( 0.55 ), float( 0 ) );
	material.emissiveNode = select( isGlass, room.xyz.mul( room.w ).mul( 5 ).mul( haze.mul( 0.7 ).oneMinus() ), color( 0x000000 ) ); // room.w = emissive weight ( 0 unlit, < 1 behind sheers ), muted by the film

	// relief: barrel-tile ridges on the roofs, recessed joints on the terrace
	// cotto and brick coursing, a soft plaster / stone grain elsewhere; glass
	// and trim stay smooth
	const grain = valueNoise( positionWorld.mul( 2.2 ) ).mul( select( isStone, float( 0.012 ), float( 0.004 ) ) );
	const roofRelief = barrel.mul( 0.014 ).mul( barrelFade ).sub( courseJoint.mul( 0.018 ) );
	const cottoRelief = cottoJoint.mul( - 0.006 );
	material.normalNode = bumpNormal( select( isRoof, roofRelief, select( isTerrace, cottoRelief, select( isBrick, brickCourse.mul( - 0.008 ), select( isGlass.or( isTrim ), float( 0 ), grain ) ) ) ) );

	return material;

}

export { HouseGenerator, createHouseMaterial, housePalette, bakeGroups, boxMatrix, metricUV, gridLine, valueNoise, valueFractal, PartId };
