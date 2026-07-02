import {
	BoxGeometry,
	BufferAttribute,
	BufferGeometry,
	ConeGeometry,
	Float32BufferAttribute,
	Group,
	IcosahedronGeometry,
	InstancedBufferAttribute,
	InstancedMesh,
	Matrix4,
	Mesh,
	Object3D,
	PlaneGeometry,
	Vector2,
	Vector3
} from 'three';

import { MeshStandardNodeMaterial } from 'three/webgpu';
import { attribute, cameraPosition, color, float, mix, normalWorld, positionView, positionWorld, normalView, saturation, smoothstep, time, vec3 } from 'three/tsl';

import { ImprovedNoise } from '../math/ImprovedNoise.js';
import { HouseGenerator, createHouseMaterial, housePalette, bakeGroups, boxMatrix, valueNoise, valueFractal, PartId } from './village/HouseGenerator.js';

const _matrix = /*@__PURE__*/ new Matrix4();
const _unitBox = /*@__PURE__*/ new BoxGeometry( 1, 1, 1 ).toNonIndexed();

/**
 * Lays a procedurally generated Ligurian fishing village — in the manner of the
 * Cinque Terre — over a coastal headland: sea cliffs, a sheltered cove with a
 * stone quay, and rows of colour-washed {@link HouseGenerator} houses stacked
 * along the contours, stepping down a ravine street to the harbour. Terraced
 * vineyard slopes rise behind, scattered with maquis scrub and cypress, and the
 * sea shades itself from a baked depth — turquoise in the cove, cobalt offshore.
 * Returns a `THREE.Group` ready to add to a scene.
 *
 * The terrain heightfield is shaped from the coastline out: a signed distance
 * to a seeded coast curve drives the cliff profile, a ravine drains to the
 * cove, and every house row stamps a flat terrace into the hill before the
 * grid is baked, so the houses' stone podiums read as the retaining walls of
 * their terraces.
 *
 * ```js
 * const village = new VillageGenerator( { seed: 1 } );
 * scene.add( village.build( materials ) );
 * ```
 */
class VillageGenerator {

	constructor( parameters = {} ) {

		this.parameters = Object.assign( {}, VillageGenerator.defaults, parameters );

		this.terrainMaterial = createTerrainMaterial();
		this.seaMaterial = null; // built per seed — its shore foam keys off the baked depth
		this.scrubMaterial = createScrubMaterial();

		this.generators = [];
		this.geometries = [];
		this.group = null;

	}

	build( materials = {} ) {

		this.dispose();

		const p = this.parameters;
		const random = createRandom( p.seed );
		const houseMaterial = materials.house || createHouseMaterial();

		const coast = buildCoast( p, random );
		const layout = buildLayout( p, random, coast );

		const group = new Group();
		group.name = 'Village';

		// --- terrain -----------------------------------------------------------

		const terrain = bakeTerrain( p, coast, layout );
		this.terrain = terrain;

		const terrainMesh = new Mesh( terrain.geometry, this.terrainMaterial );
		terrainMesh.castShadow = terrainMesh.receiveShadow = true;
		terrainMesh.name = 'Terrain';
		group.add( terrainMesh );
		this.geometries.push( terrain.geometry );

		// --- houses --------------------------------------------------------------

		for ( const house of layout.houses ) {

			const generator = new HouseGenerator( house.parameters, houseMaterial );
			const mesh = generator.build();
			mesh.position.set( house.x, house.base, house.z );
			mesh.rotation.y = house.rotationY;
			mesh.castShadow = mesh.receiveShadow = true;

			group.add( mesh );
			this.generators.push( generator );

		}

		// --- masonry: quay, slipway, breakwater, shore boulders, campanile ------

		const masonry = buildMasonry( p, random, coast, layout );
		const masonryMesh = new Mesh( masonry, houseMaterial );
		masonryMesh.castShadow = masonryMesh.receiveShadow = true;
		masonryMesh.name = 'Masonry';
		group.add( masonryMesh );
		this.geometries.push( masonry );

		// --- sea -----------------------------------------------------------------

		const sea = bakeSea( p, coast );
		if ( this.seaMaterial === null ) this.seaMaterial = createSeaMaterial();

		const seaMesh = new Mesh( sea, this.seaMaterial );
		seaMesh.receiveShadow = true;
		seaMesh.name = 'Sea';
		group.add( seaMesh );
		this.geometries.push( sea );

		// --- vegetation ----------------------------------------------------------

		const scrub = buildScrub( p, random, coast, terrain, layout, this.scrubMaterial );
		group.add( scrub );
		this.geometries.push( scrub.geometry );

		this.group = group;

		return group;

	}

	// world-space terrain height at ( x, z ), bilinearly interpolated from the baked grid
	sampleHeight( x, z ) {

		const p = this.parameters;
		const N = p.segments + 1;
		const half = p.size / 2;

		const fx = Math.max( 0, Math.min( p.segments, ( x + half ) / p.size * p.segments ) );
		const fz = Math.max( 0, Math.min( p.segments, ( z + half ) / p.size * p.segments ) );

		const ix = Math.min( N - 2, Math.floor( fx ) );
		const iz = Math.min( N - 2, Math.floor( fz ) );
		const tx = fx - ix;
		const tz = fz - iz;

		const h = this.terrain.heights;
		const h00 = h[ iz * N + ix ];
		const h10 = h[ iz * N + ix + 1 ];
		const h01 = h[ ( iz + 1 ) * N + ix ];
		const h11 = h[ ( iz + 1 ) * N + ix + 1 ];

		return ( h00 * ( 1 - tx ) + h10 * tx ) * ( 1 - tz ) + ( h01 * ( 1 - tx ) + h11 * tx ) * tz;

	}

	dispose() {

		for ( const generator of this.generators ) generator.dispose();
		this.generators.length = 0;

		for ( const geometry of this.geometries ) geometry.dispose();
		this.geometries.length = 0;

		this.group = null;

	}

}

VillageGenerator.defaults = {
	seed: 1,
	size: 520, // world units across the square terrain patch
	segments: 384, // grid quads per side — ~1.35 m cells, enough for the terrace stamps
	scrubCount: 3200 // maquis blobs scattered over the slopes ( one instanced draw )
};

// deterministic PRNG ( mulberry32 ), matching the other generators
function createRandom( seed ) {

	let s = ( seed >>> 0 ) || 1;

	return function () {

		s = ( s + 0x6D2B79F5 ) | 0;
		let t = Math.imul( s ^ ( s >>> 15 ), 1 | s );
		t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

function gauss( u ) {

	return Math.exp( - u * u );

}

function smoothBlend( edge0, edge1, x ) {

	const t = Math.max( 0, Math.min( 1, ( x - edge0 ) / ( edge1 - edge0 ) ) );
	return t * t * ( 3 - 2 * t );

}

// --- coast ------------------------------------------------------------------

/**
 * The seeded shape of the shoreline and everything derived from it. The coast is
 * a curve z( x ): a headland bulges seaward, the cove notches inland, and noise
 * meanders the line. Land lies at z below the curve; the sea above it.
 */
function buildCoast( p, random ) {

	const perlin = new ImprovedNoise();

	// ImprovedNoise's permutation is fixed, so the seed shifts the sample window
	const offset = random() * 256;
	const slice = random() * 256;

	// the cove drifts a little per seed; the headland anchors the west
	const coveX = 30 + random() * 12;
	const coveWidth = 30 + random() * 6;

	const coastZ = ( x ) => 30
		+ 58 * gauss( ( x + 70 ) / 78 ) // the headland, shouldering into the sea
		- 62 * gauss( ( x - coveX ) / coveWidth ) // the cove, notched into it
		+ 7 * perlin.noise( x * 0.02 + offset, 0.5, slice ); // a meander so the line never reads drawn

	// the cliff-top height along the coast: highest on the headland, dropping to
	// the harbour at the cove so the village can step down to the water
	const cliffTop = ( x ) => Math.max( 3,
		14
		+ 20 * gauss( ( x + 70 ) / 95 )
		- 12 * gauss( ( x - coveX ) / ( coveWidth * 1.6 ) )
		+ 4 * perlin.noise( x * 0.013 + offset, 7.5, slice ) );

	// the coastline sampled fine enough for a true distance query
	const step = 2;
	const count = Math.ceil( p.size / step ) + 1;
	const points = [];

	for ( let i = 0; i < count; i ++ ) {

		const x = - p.size / 2 + i * step;
		points.push( x, coastZ( x ) );

	}

	// signed distance to the coastline: positive inland, negative at sea. a coarse
	// scan then a fine window keeps the search cheap across the whole bake.
	const coastDistance = ( x, z ) => {

		let best = Infinity;
		let bestI = 0;

		for ( let i = 0; i < count; i += 4 ) {

			const dx = points[ i * 2 ] - x;
			const dz = points[ i * 2 + 1 ] - z;
			const d = dx * dx + dz * dz;
			if ( d < best ) {

				best = d;
				bestI = i;

			}

		}

		for ( let i = Math.max( 0, bestI - 4 ); i < Math.min( count, bestI + 5 ); i ++ ) {

			const dx = points[ i * 2 ] - x;
			const dz = points[ i * 2 + 1 ] - z;
			const d = dx * dx + dz * dz;
			if ( d < best ) best = d;

		}

		return z < coastZ( x ) ? Math.sqrt( best ) : - Math.sqrt( best );

	};

	// the ravine: a valley draining from the saddle down to the cove head — the
	// amphitheatre every terraced row wraps around
	const ravineA = new Vector2( coveX, - 26 );
	const ravineDir = new Vector2( - 0.3, - 1 ).normalize();

	const ravine = ( x, z ) => {

		const qx = x - ravineA.x;
		const qz = z - ravineA.y;
		const t = Math.max( 0, qx * ravineDir.x + qz * ravineDir.y );
		const rx = qx - ravineDir.x * t;
		const rz = qz - ravineDir.y * t;
		const r = Math.hypot( rx, rz );

		return { t, r };

	};

	// the base heightfield, before terraces and stamps
	const baseHeight = ( x, z ) => {

		const d = coastDistance( x, z );

		if ( d <= 0 ) {

			// the seabed: a shelf dropping offshore, held to a boat-shallow basin
			// inside the cove so the water shades turquoise over it
			let h = - 1 - Math.min( 15, - d * 0.34 );
			const cove = gauss( ( x - coveX ) / ( coveWidth * 0.85 ) ) * smoothBlend( 40, 0, z );
			h = h * ( 1 - cove ) - 3 * cove;

			return Math.min( - 0.7, h + perlin.noise( x * 0.05 + offset, z * 0.05, slice + 9 ) * 0.6 );

		}

		// the cliff: a near-vertical rise off the waterline up to the cliff top,
		// then the hillside climbing away toward a back ridge whose crest
		// wanders per seed, so the skyline never reads as one drawn cone
		let h = cliffTop( x ) * Math.pow( smoothBlend( 0, 15, d ), 0.6 );
		h += Math.max( 0, d - 18 ) * 0.36;
		h += ( 40 + 26 * perlin.noise( x * 0.007 + offset, 3.1, slice + 5 ) ) * smoothBlend( 130, 250, d );

		// the ravine carves the amphitheatre; its floor stays gentle for the street
		const { t, r } = ravine( x, z );
		const width = 26 + t * 0.14;
		h -= ( 13 + t * 0.12 ) * gauss( r / width ) * smoothBlend( 2, 24, d );

		// fractal relief, quiet at the shore and swelling inland; damped along the
		// ravine floor so the street bed stays clean
		const amp = ( 1.5 + 9 * smoothBlend( 10, 90, d ) ) * ( 1 - 0.7 * gauss( r / width ) );
		let noise = 0, freq = 0.008, a = 1, norm = 0;

		for ( let o = 0; o < 4; o ++ ) {

			noise += a * perlin.noise( x * freq + offset, z * freq + offset * 0.7, slice + o * 1.7 );
			norm += a; a *= 0.5; freq *= 2.1;

		}

		h += noise / norm * amp * 2;

		// the land sinks back into the sea toward the patch borders, so the
		// headland reads as a promontory from every orbit, never a cut slab
		const fade = Math.max(
			smoothBlend( 208, 254, Math.abs( x ) ),
			smoothBlend( - 192, - 250, z ) );

		return h * ( 1 - fade ) - 8 * fade;

	};

	return { coastZ, cliffTop, coastDistance, baseHeight, ravine, points, count, coveX, coveWidth, perlin, offset, slice };

}

// --- layout -------------------------------------------------------------------

// a polyline with arc-length sampling, for walking house rows along a path
class Polyline {

	constructor( points ) {

		this.points = points;
		this.lengths = [ 0 ];

		for ( let i = 1; i < points.length; i ++ ) {

			this.lengths.push( this.lengths[ i - 1 ] + points[ i ].distanceTo( points[ i - 1 ] ) );

		}

		this.length = this.lengths[ this.lengths.length - 1 ];

	}

	// position and unit tangent at arc length s
	sample( s ) {

		const L = this.lengths;
		let i = 1;
		while ( i < L.length - 1 && L[ i ] < s ) i ++;

		const a = this.points[ i - 1 ], b = this.points[ i ];
		const t = ( s - L[ i - 1 ] ) / Math.max( 1e-6, L[ i ] - L[ i - 1 ] );

		const tx = b.x - a.x, tz = b.y - a.y;
		const inv = 1 / Math.max( 1e-6, Math.hypot( tx, tz ) );

		return {
			x: a.x + ( b.x - a.x ) * t,
			z: a.y + ( b.y - a.y ) * t,
			tx: tx * inv,
			tz: tz * inv
		};

	}

}

// the coastline offset inland by `offset`, over the given x window — the natural
// path for a row of houses that hugs the shore
function offsetCoastPath( coast, x0, x1, offset ) {

	const points = [];
	const step = 4;

	for ( let x = x0; x <= x1; x += step ) {

		// inland normal from the coast tangent ( dz/dx )
		const dz = ( coast.coastZ( x + 2 ) - coast.coastZ( x - 2 ) ) / 4;
		const inv = 1 / Math.hypot( dz, 1 );
		points.push( new Vector2( x + dz * inv * offset, coast.coastZ( x ) - inv * offset ) );

	}

	// a light smoothing pass, so the noise meander doesn't kink the row
	for ( let pass = 0; pass < 2; pass ++ ) {

		for ( let i = 1; i < points.length - 1; i ++ ) {

			points[ i ].x = ( points[ i - 1 ].x + points[ i ].x * 2 + points[ i + 1 ].x ) / 4;
			points[ i ].y = ( points[ i - 1 ].y + points[ i ].y * 2 + points[ i + 1 ].y ) / 4;

		}

	}

	return new Polyline( points );

}

// walks houses along a path: contiguous facades, party walls shared, each house
// snapped to its own base height — the stepped rows of the coast
function layoutRow( layout, random, path, options ) {

	const { elevation, facing = 1, depth, floorsMin = 2, floorsMax = 4, margin = 2, gap = 0 } = options;

	let s = margin;
	let previous = null;

	while ( s < path.length - margin - 5 ) {

		const width = 5.5 + random() * 4;
		if ( s + width > path.length - margin ) break;

		const at = path.sample( s + width / 2 );

		// the outward facade normal: the path tangent spun toward `facing`
		const nx = at.tz * facing;
		const nz = - at.tx * facing;

		const base = Math.round( elevation( s + width / 2 ) * 2 ) / 2;

		let floors = floorsMin + Math.floor( random() * ( floorsMax - floorsMin + 1 ) );
		if ( previous && floors === previous.floors && base === previous.base ) floors = floors === floorsMax ? floors - 1 : floors + 1;

		// paint pick avoiding the neighbour's, so no two attached houses match
		let paint = Math.floor( random() * housePalette.length );
		if ( previous && paint === previous.paint ) paint = ( paint + 3 + Math.floor( random() * 5 ) ) % housePalette.length;

		const d = depth + ( random() - 0.5 );

		// footprint centre sits half a depth behind the facade line
		const x = at.x - nx * d / 2;
		const z = at.z - nz * d / 2;

		const house = {
			x, z, base,
			rotationY: Math.atan2( nx, nz ),
			floors,
			paint,
			parameters: {
				seed: Math.floor( random() * 100000 ),
				width: width - 0.02,
				depth: d,
				floors,
				paint,
				podiumDepth: 10,
				leftExposed: previous === null,
				rightExposed: false // set true on whichever house ends the row
			}
		};

		layout.houses.push( house );

		// the terrace: a flat shelf under the house and its walk, feathered into
		// the hill, with a hair of drop so podium tops never poke through
		layout.stamps.push( {
			cx: x + nx * 1.2, cz: z + nz * 1.2,
			hw: width / 2 + 1.5, hd: d / 2 + 3.4,
			cos: nz, sin: nx,
			feather: 6,
			h: base - 0.05
		} );

		previous = house;
		s += width + gap;

	}

	if ( previous ) previous.parameters.rightExposed = true;

}

// stamps a walkable lane along a path — the streets and the harbour front
function layoutLane( layout, path, elevation, radius ) {

	for ( let s = 0; s < path.length; s += 3 ) {

		const at = path.sample( s );
		layout.stamps.push( { cx: at.x, cz: at.z, r: radius, feather: 5, h: elevation( s ) } );

	}

}

/**
 * Places every row of the village: the stacked cliff-front rows over the
 * headland, the smaller cluster across the cove, the street pair stepping down
 * the ravine to the harbour, and the harbour front itself — plus the terrace
 * and lane stamps that flatten the hill under all of it.
 */
function buildLayout( p, random, coast ) {

	const layout = { houses: [], stamps: [], street: null };

	const coveX = coast.coveX;

	// headland rows: hugging the coast west of the cove, each terrace pinned just
	// above the grade under it, so the cluster steps up the hill and cascades
	// down toward the harbour where the ravine cuts through

	const grade = ( path ) => ( s ) => {

		const at = path.sample( s );
		return Math.max( 4.5, coast.baseHeight( at.x, at.z ) + 1 );

	};

	const westRows = 4 + Math.floor( random() * 2 );

	for ( let k = 0; k < westRows; k ++ ) {

		const path = offsetCoastPath( coast, - 122 + k * 12 + random() * 6, coveX - 24 - k * 7, 7.5 + k * 11 );

		layoutRow( layout, random, path, {
			elevation: grade( path ),
			facing: - 1,
			depth: 8.5 + random(),
			floorsMin: 2 + ( k === 0 ? 1 : 0 ), // the front row stands tallest against the sea
			floorsMax: 4
		} );

	}

	// the far side of the cove: a smaller, lower cluster

	const eastRows = 2 + Math.floor( random() * 2 );

	for ( let k = 0; k < eastRows; k ++ ) {

		const path = offsetCoastPath( coast, coveX + 22 + k * 5, coveX + 64 - k * 4, 6.5 + k * 10.5 );

		layoutRow( layout, random, path, {
			elevation: grade( path ),
			facing: - 1,
			depth: 8 + random(),
			floorsMin: 2,
			floorsMax: 3
		} );

	}

	// the street: a pair of facing rows stepping down the ravine to the harbour.
	// its profile follows the valley floor, smoothed monotonic so it only descends.

	const streetFrom = new Vector2( coveX - 0.5, - 28 );
	const dir = new Vector2( - 0.3, - 1 ).normalize();
	const streetLength = 92 + random() * 20;

	const axisPoints = [];
	for ( let s = 0; s <= streetLength; s += 6 ) {

		// a slight seeded bow, so the street reads as grown rather than drawn
		const bow = Math.sin( s / streetLength * Math.PI ) * ( 6 + random() * 2 ) * 0.5;
		axisPoints.push( new Vector2(
			streetFrom.x + dir.x * s - dir.y * bow,
			streetFrom.y + dir.y * s + dir.x * bow
		) );

	}

	const axis = new Polyline( axisPoints.reverse() ); // sampled top → harbour

	const profile = [];
	let running = Infinity;

	for ( let s = 0; s <= axis.length; s += 3 ) {

		const at = axis.sample( s );
		running = Math.min( running, Math.max( 5, coast.baseHeight( at.x, at.z ) ) );
		profile.push( running );

	}

	const streetElevation = ( s ) => {

		const i = Math.min( profile.length - 1, Math.max( 0, s / 3 ) );
		const i0 = Math.floor( i );
		const t = i - i0;
		return profile[ i0 ] * ( 1 - t ) + profile[ Math.min( profile.length - 1, i0 + 1 ) ] * t;

	};

	layoutLane( layout, axis, streetElevation, 4.5 );
	layout.street = { axis, elevation: streetElevation };

	for ( const side of [ - 1, 1 ] ) {

		const offsetPoints = axisPoints.map( ( q, i ) => {

			const at = axis.sample( axis.lengths[ i ] );
			return new Vector2( q.x + at.tz * side * 5.2, q.y - at.tx * side * 5.2 );

		} );

		layoutRow( layout, random, new Polyline( offsetPoints ), {
			elevation: ( s ) => streetElevation( s ) + 0.1,
			facing: - side, // both rows face the street between them
			depth: 7.5 + random(),
			floorsMin: 2,
			floorsMax: 4,
			margin: 4
		} );

	}

	// the harbour front: a short bright row at the cove head, facing the water

	const harbourZ = coast.coastZ( coveX ) - 7;
	const harbourPath = new Polyline( [
		new Vector2( coveX - 20, harbourZ - 2 ),
		new Vector2( coveX, harbourZ - 3.5 ),
		new Vector2( coveX + 20, harbourZ - 2 )
	] );

	layoutRow( layout, random, harbourPath, {
		elevation: () => 4.5,
		facing: - 1,
		depth: 8 + random(),
		floorsMin: 3, // the harbour houses are the tall postcard ones
		floorsMax: 5,
		margin: 1
	} );

	layout.harbour = { x: coveX, z: harbourZ };

	// the quay apron in front of the harbour row, and its stamp so the ground
	// under the paving holds still

	layout.stamps.push( { cx: coveX, cz: harbourZ + 6, hw: 21, hd: 6, cos: 1, sin: 0, feather: 5, h: 2.1 } );

	// the campanile watches from the top of the headland cluster

	const bell = offsetCoastPath( coast, - 80, - 40, 12 + westRows * 11 ).sample( 12 );
	layout.campanile = { x: bell.x, z: bell.z, base: Math.max( 6, coast.baseHeight( bell.x, bell.z ) + 1 ) };

	layout.stamps.push( { cx: layout.campanile.x, cz: layout.campanile.z, r: 8, feather: 6, h: layout.campanile.base - 0.05 } );

	return layout;

}

// --- terrain -------------------------------------------------------------------

/**
 * Bakes the heightfield to a grid: the coast-driven base, vineyard terracettes
 * quantized into the mid slopes, then every village stamp pressed in last so
 * streets and house terraces stay dead flat. The stamp weight is kept as a
 * `paved` vertex attribute for the material, and the grid is kept for sampling.
 */
function bakeTerrain( p, coast, layout ) {

	const N = p.segments + 1;
	const half = p.size / 2;
	const cell = p.size / p.segments;

	const heights = new Float32Array( N * N );
	const paved = new Float32Array( N * N );

	const coord = new Array( N );
	for ( let i = 0; i < N; i ++ ) coord[ i ] = i / p.segments * p.size - half;

	for ( let iz = 0; iz < N; iz ++ ) {

		for ( let ix = 0; ix < N; ix ++ ) {

			const x = coord[ ix ], z = coord[ iz ];
			let h = coast.baseHeight( x, z );

			// vineyard terracettes: the hillside quantized into dry-stone steps,
			// feathered by noise so the banding never reads mechanical
			if ( h > 10 && h < 110 ) {

				const band = smoothBlend( 10, 18, h ) * smoothBlend( 110, 85, h );
				const m = band * ( 0.7 + 0.6 * coast.perlin.noise( x * 0.025 + coast.offset, z * 0.025, coast.slice + 31 ) );
				h += ( Math.round( h / 4.2 ) * 4.2 - h ) * Math.max( 0, Math.min( 0.95, m ) );

			}

			heights[ iz * N + ix ] = h;

		}

	}

	// press the stamps in: rectangles under houses, discs along lanes

	for ( const s of layout.stamps ) {

		const reach = ( s.r || Math.hypot( s.hw, s.hd ) ) + s.feather;
		const ix0 = Math.max( 0, Math.floor( ( s.cx - reach + half ) / cell ) );
		const ix1 = Math.min( N - 1, Math.ceil( ( s.cx + reach + half ) / cell ) );
		const iz0 = Math.max( 0, Math.floor( ( s.cz - reach + half ) / cell ) );
		const iz1 = Math.min( N - 1, Math.ceil( ( s.cz + reach + half ) / cell ) );

		for ( let iz = iz0; iz <= iz1; iz ++ ) {

			for ( let ix = ix0; ix <= ix1; ix ++ ) {

				const x = coord[ ix ] - s.cx;
				const z = coord[ iz ] - s.cz;

				let outside;

				if ( s.r ) {

					outside = Math.hypot( x, z ) - s.r;

				} else {

					// rotate into the stamp's frame; distance outside its rectangle
					const u = x * s.cos + z * s.sin;
					const v = - x * s.sin + z * s.cos;
					outside = Math.max( Math.abs( u ) - s.hw, Math.abs( v ) - s.hd );

				}

				const w = smoothBlend( s.feather, 0, Math.max( 0, outside ) );
				if ( w <= 0 ) continue;

				const i = iz * N + ix;
				heights[ i ] += ( s.h - heights[ i ] ) * w;
				if ( w > paved[ i ] ) paved[ i ] = w;

			}

		}

	}

	// lay the grid out flat in the XZ plane, diagonals alternated so the mesh
	// reads as diamonds rather than a one-way grain

	const positions = new Float32Array( N * N * 3 );

	for ( let iz = 0; iz < N; iz ++ ) {

		for ( let ix = 0; ix < N; ix ++ ) {

			const o = iz * N + ix;
			positions[ o * 3 ] = coord[ ix ];
			positions[ o * 3 + 1 ] = heights[ o ];
			positions[ o * 3 + 2 ] = coord[ iz ];

		}

	}

	const indices = [];

	for ( let iz = 0; iz < p.segments; iz ++ ) {

		for ( let ix = 0; ix < p.segments; ix ++ ) {

			const a = iz * N + ix, b = a + 1, c = a + N, d = c + 1;

			if ( ( ix + iz ) % 2 === 0 ) indices.push( a, c, b, b, c, d );
			else indices.push( a, c, d, a, d, b );

		}

	}

	const geometry = new BufferGeometry();
	geometry.setAttribute( 'position', new Float32BufferAttribute( positions, 3 ) );
	geometry.setAttribute( 'paved', new Float32BufferAttribute( paved, 1 ) );
	geometry.setIndex( indices );
	geometry.computeVertexNormals();

	return { geometry, heights, paved, gridSize: N };

}

// --- masonry ---------------------------------------------------------------------

// the village's stonework, baked into one geometry with the house material's part
// codes: the quay and slipway, a breakwater arm, boulders along the waterline and
// the campanile that crowns the cluster
function buildMasonry( p, random, coast, layout ) {

	const stone = [];
	const walls = [];
	const trims = [];
	const glass = [];
	const boulders = [];

	// the quay: a stone apron holding the harbour front out of the swell

	const hx = layout.harbour.x, hz = layout.harbour.z;
	stone.push( boxMatrix( hx, - 0.6, hz + 6, 42, 5.6, 12 ) );
	stone.push( boxMatrix( hx, 2.28, hz + 6, 42, 0.16, 12 ) ); // a paving lip proud of the wall face

	// the slipway, dipping from the quay into the cove

	_matrix.makeRotationX( 0.32 );
	stone.push( new Matrix4()
		.makeTranslation( hx + 6, 0.8, hz + 13.5 )
		.multiply( _matrix )
		.scale( new Vector3( 5, 0.5, 9 ) ) );

	// the breakwater: an arm of piled blocks sheltering the cove mouth from the east

	const armX = coast.coveX + coast.coveWidth * 0.72;
	const armZ0 = coast.coastZ( armX ) - 2;

	for ( let i = 0; i < 9; i ++ ) {

		const t = i / 8;
		boulders.push( rockMatrix( random,
			armX - t * 6 - random() * 2,
			- 0.8 - t * 0.5,
			armZ0 + t * 18,
			2.2 + random() * 1.6 ) );

	}

	// boulders along the waterline wherever the cliffs meet the sea

	for ( let x = - 200; x < 200; x += 5 ) {

		if ( Math.abs( x - coast.coveX ) < coast.coveWidth * 0.9 ) continue; // the cove mouth stays clear
		if ( random() < 0.45 ) continue;

		const z = coast.coastZ( x );
		const along = ( random() - 0.5 ) * 4;

		boulders.push( rockMatrix( random,
			x + along,
			- 0.6 - random() * 0.8,
			z + ( random() - 0.5 ) * 5,
			1 + random() * random() * 2.4 ) );

	}

	// the campanile: a plastered shaft, an open belfry and a pyramid cap — the
	// landmark that fixes the skyline, as San Lorenzo's tower does at Manarola

	const c = layout.campanile;
	const base = c.base;
	const shaft = 12 + random() * 2;

	stone.push( boxMatrix( c.x, base - 4, c.z, 4.6, 8, 4.6 ) ); // its own stone footing
	walls.push( boxMatrix( c.x, base + shaft / 2, c.z, 3.6, shaft, 3.6 ) );
	trims.push( boxMatrix( c.x, base + shaft + 0.06, c.z, 4.1, 0.24, 4.1 ) );

	// belfry: four corner piers around a dark open chamber
	const bh = 2.6;
	for ( const sx of [ - 1, 1 ] ) {

		for ( const sz of [ - 1, 1 ] ) {

			walls.push( boxMatrix( c.x + sx * 1.45, base + shaft + 0.18 + bh / 2, c.z + sz * 1.45, 0.7, bh, 0.7 ) );

		}

	}

	glass.push( boxMatrix( c.x, base + shaft + 0.18 + bh / 2, c.z, 2.4, bh - 0.3, 2.4 ) );
	trims.push( boxMatrix( c.x, base + shaft + bh + 0.3, c.z, 4.2, 0.3, 4.2 ) );

	// the pyramid cap: a four-sided cone, spun so its faces align with the shaft
	const cap = new ConeGeometry( 2.9, 2.4, 4, 1 ).rotateY( Math.PI / 4 ).toNonIndexed();

	const groups = [
		{ geometry: new IcosahedronGeometry( 1, 1 ), matrices: boulders, partId: PartId.STONE },
		{ geometry: _unitBox, matrices: stone, partId: PartId.STONE },
		{ geometry: _unitBox, matrices: walls, partId: PartId.WALL },
		{ geometry: _unitBox, matrices: trims, partId: PartId.TRIM },
		{ geometry: _unitBox, matrices: glass, partId: PartId.GLASS },
		{ geometry: cap, matrices: [ new Matrix4().makeTranslation( c.x, base + shaft + bh + 1.65, c.z ) ], partId: PartId.ROOF, rigid: true }
	];

	// the campanile plasters in the palette's cream
	return bakeGroups( groups, housePalette.length - 3, random() );

}

// a coarse icosphere spun and squashed per boulder, so the same hull reads as
// varied shore rock
function rockMatrix( random, x, y, z, size ) {

	return new Matrix4()
		.makeTranslation( x, y, z )
		.multiply( _matrix.makeRotationY( random() * Math.PI * 2 ) )
		.scale( new Vector3( size * ( 0.7 + random() * 0.6 ), size * ( 0.5 + random() * 0.5 ), size * ( 0.7 + random() * 0.6 ) ) );

}

// --- sea ---------------------------------------------------------------------------

// a broad grid carrying the seabed depth per vertex, so the material can shade
// the shallows and break foam along the shore without ever sampling the terrain
function bakeSea( p, coast ) {

	const span = 2600;
	const segments = 300;

	const geometry = new PlaneGeometry( span, span, segments, segments );
	geometry.rotateX( - Math.PI / 2 );

	const position = geometry.attributes.position;
	const depths = new Float32Array( position.count );
	const half = p.size / 2;

	for ( let i = 0; i < position.count; i ++ ) {

		const x = position.getX( i );
		const z = position.getZ( i );

		if ( x > - half && x < half && z > - half && z < half ) {

			depths[ i ] = Math.max( 0.15, - coast.baseHeight( x, z ) );

		} else {

			// open water past the terrain patch, deepening toward the horizon
			depths[ i ] = 18 + Math.min( 30, ( Math.max( Math.abs( x ), Math.abs( z ) ) - half ) * 0.05 );

		}

	}

	geometry.setAttribute( 'seaDepth', new BufferAttribute( depths, 1 ) );
	geometry.computeBoundingSphere();

	return geometry;

}

// derivative-based bump ( surface-gradient method ), for procedural world-space
// height fields — see the note in HouseGenerator
function bumpNormal( height ) {

	const dpdx = positionView.dFdx();
	const dpdy = positionView.dFdy();
	const r1 = dpdy.cross( normalView );
	const r2 = normalView.cross( dpdx );
	const det = dpdx.dot( r1 );
	const grad = det.sign().mul( height.dFdx().mul( r1 ).add( height.dFdy().mul( r2 ) ) );

	return det.abs().mul( normalView ).sub( grad ).normalize();

}

/**
 * The Ligurian sea: cobalt offshore, flaring turquoise over the shallow cove,
 * with wind-ruffled waves and a fringe of animated foam along the shore. Depth
 * comes baked per vertex ( see the sea grid ), so the shading costs no terrain
 * lookups.
 */
function createSeaMaterial() {

	const depth = attribute( 'seaDepth', 'float' );

	// wind-ruffled surface: three octaves of drifting value noise become the
	// wave height; its screen-space gradient perturbs the normal. the ruffle
	// eases off with distance so the horizon can't shimmer.
	const calm = smoothstep( 900, 150, positionWorld.distance( cameraPosition ) );
	const waves = valueNoise( vec3( positionWorld.x.mul( 0.08 ), positionWorld.z.mul( 0.08 ), time.mul( 0.22 ) ) )
		.add( valueNoise( vec3( positionWorld.x.mul( 0.21 ).add( time.mul( 0.1 ) ), positionWorld.z.mul( 0.19 ), time.mul( 0.34 ) ) ).mul( 0.5 ) )
		.add( valueNoise( vec3( positionWorld.x.mul( 0.55 ), positionWorld.z.mul( 0.5 ).sub( time.mul( 0.16 ) ), time.mul( 0.5 ) ) ).mul( 0.25 ) )
		.mul( calm );

	// colour by depth: deep cobalt swallows the seabed offshore, the shallows
	// flare bright turquoise where light still reaches the sand. the flare is
	// deliberately strong — at grazing angles fresnel reflection mutes it.
	let water = mix( color( 0x093052 ), color( 0x15719f ), depth.mul( - 0.09 ).exp() );
	water = mix( water, color( 0x53dcc8 ), depth.mul( - 0.22 ).exp() );

	// foam: a breathing fringe where the water thins against the shore
	const surge = valueNoise( vec3( positionWorld.x.mul( 0.35 ), positionWorld.z.mul( 0.35 ), time.mul( 0.4 ) ) ).mul( 0.5 ).add( 0.5 );
	const fringe = smoothstep( 1.7, 0.1, depth.add( surge ) );
	const foam = fringe.mul( surge.mul( 0.6 ).add( 0.4 ) );

	const material = new MeshStandardNodeMaterial();
	material.colorNode = mix( water, color( 0xe9f6f2 ), foam );
	material.roughnessNode = mix( float( 0.1 ), float( 0.55 ), foam );
	material.metalnessNode = float( 0 );
	material.normalNode = bumpNormal( waves.mul( 0.3 ).mul( smoothstep( 0.4, 3, depth ).mul( 0.8 ).add( 0.2 ) ) ); // the swell settles in the sheltered shallows

	return material;

}

// --- terrain material ---------------------------------------------------------------

/**
 * Altitude- and slope-based shading for the headland, all in TSL: stratified sea
 * cliffs, maquis scrub and dry summer grass above, stone showing on every
 * terracette riser, wave-washed rock at the waterline — and the stamped village
 * ground reads back its `paved` weight to become worn stone paving.
 */
function createTerrainMaterial() {

	const material = new MeshStandardNodeMaterial();
	material.metalness = 0;

	const distance = positionWorld.distance( cameraPosition );

	const flatness = normalWorld.y.clamp();
	const steep = flatness.oneMinus();
	const height = positionWorld.y;

	const grain = valueNoise( positionWorld.mul( 0.16 ) );
	const macro = valueNoise( positionWorld.mul( 0.02 ) );

	// stratified cliff rock: warm grey-brown bedding planes at two frequencies,
	// wobbled by noise — the tilted sandstone of the riviera
	const bandA = height.mul( 0.55 ).add( grain.mul( 2.2 ) ).add( macro.mul( 3 ) ).sin();
	const bandB = height.mul( 1.6 ).add( grain.mul( 3 ) ).sin();
	const strata = bandA.mul( 0.6 ).add( bandB.mul( 0.4 ) ).mul( 0.5 ).add( 0.5 );
	const rock = mix( color( 0x4a4336 ), color( 0x8a7d67 ), strata );

	// the vegetated hill: dry maquis scrub drifting through summer-scorched grass
	const scrub = mix( color( 0x4d5c33 ), color( 0x6d7040 ), grain.mul( 0.5 ).add( 0.5 ) );
	const grass = mix( color( 0x97854e ), color( 0xb0a060 ), macro.mul( 0.5 ).add( 0.5 ) );
	let surface = mix( scrub, grass, smoothstep( - 0.2, 0.6, macro ).mul( 0.75 ) );

	// dry-stone shows on every steep face — sea cliffs and terracette risers
	// alike, which is what draws the vineyard contours on the hill
	surface = mix( surface, rock, smoothstep( 0.28, 0.5, steep ) );

	// wave-washed base of the cliffs: darker, wet, with a band of algae at the swell
	surface = mix( surface, color( 0x453f35 ), smoothstep( 2.4, 0.6, height ).mul( 0.65 ) );
	surface = mix( surface, color( 0x39514a ), smoothstep( 1.2, 0.2, height ).mul( 0.6 ) );

	// the stamped village ground: worn stone paving on the flats
	const paved = attribute( 'paved', 'float' );
	const paving = mix( color( 0x9a927e ), color( 0xb0a58e ), grain.mul( 0.5 ).add( 0.5 ) );
	surface = mix( surface, paving, smoothstep( 0.35, 0.75, paved ).mul( flatness ) );

	// macro drift then a fine mottle, so no band reads as a flat fill
	surface = surface.mul( macro.mul( 0.15 ).add( 0.92 ) );
	surface = surface.mul( grain.mul( 0.08 ).add( 0.96 ) );

	// aerial perspective: the far ridges dry out and lift into the haze
	const aerial = smoothstep( 380, 1100, distance );
	surface = saturation( surface, aerial.oneMinus().mul( 0.5 ).add( 0.5 ) );
	surface = mix( surface, color( 0xc9cfd4 ), aerial.mul( 0.55 ) );

	material.colorNode = surface;
	material.roughnessNode = float( 0.95 );

	// relief, strongest on the rock faces and faded with distance so it can't
	// shimmer into the haze
	const relief = valueFractal( positionWorld.mul( 0.35 ), 2 )
		.mul( smoothstep( 340, 40, distance ) )
		.mul( mix( float( 0.2 ), float( 0.7 ), steep ) );
	material.normalNode = bumpNormal( relief );

	return material;

}

// --- vegetation -------------------------------------------------------------------

// scatters maquis scrub and cypress over the slopes: one instanced draw of a
// squashed icosphere, stretched tall for the cypress and tinted per instance
function buildScrub( p, random, coast, terrain, layout, material ) {

	const count = p.scrubCount;
	const half = p.size / 2;
	const N = terrain.gridSize;
	const cell = p.size / p.segments;

	const sampleGrid = ( grid, x, z ) => {

		const ix = Math.max( 0, Math.min( N - 1, Math.round( ( x + half ) / cell ) ) );
		const iz = Math.max( 0, Math.min( N - 1, Math.round( ( z + half ) / cell ) ) );
		return grid[ iz * N + ix ];

	};

	const geometry = blobGeometry();
	const mesh = new InstancedMesh( geometry, material, count );
	mesh.castShadow = mesh.receiveShadow = true;

	const shade = new Float32Array( count );
	const dummy = new Object3D();
	let placed = 0;
	let attempts = 0;

	while ( placed < count && attempts < count * 20 ) {

		attempts ++;

		const x = ( random() - 0.5 ) * p.size;
		const z = ( random() - 0.5 ) * p.size;

		const y = sampleGrid( terrain.heights, x, z );
		if ( y < 2 || y > 115 ) continue;
		if ( sampleGrid( terrain.paved, x, z ) > 0.12 ) continue; // stay off the streets and terraces

		// keep off the sheerest cliff faces
		const rise = Math.abs( sampleGrid( terrain.heights, x + cell, z ) - sampleGrid( terrain.heights, x - cell, z ) )
			+ Math.abs( sampleGrid( terrain.heights, x, z + cell ) - sampleGrid( terrain.heights, x, z - cell ) );
		if ( rise > cell * 3.4 ) continue;

		// clumped, not even: a squared low-frequency mask opens real clearings
		// between thickets instead of an even peppering
		const density = coast.perlin.noise( x * 0.035 + coast.offset, z * 0.035, coast.slice + 77 ) * 0.5 + 0.5;
		if ( random() > density * density * 1.6 ) continue;

		const cypress = random() < 0.025 && y < 70;
		const s = 1 + random() * random() * 1.6;

		dummy.position.set( x, y - 0.4, z );
		dummy.rotation.set( ( random() - 0.5 ) * 0.15, random() * Math.PI * 2, ( random() - 0.5 ) * 0.15 );

		if ( cypress ) dummy.scale.set( s * 0.35, s * ( 3.2 + random() * 1.6 ), s * 0.35 );
		else dummy.scale.set( s * ( 0.9 + random() * 0.5 ), s * ( 0.8 + random() * 0.5 ), s * ( 0.9 + random() * 0.5 ) );

		dummy.updateMatrix();
		mesh.setMatrixAt( placed, dummy.matrix );
		shade[ placed ] = cypress ? 1 : random();

		placed ++;

	}

	mesh.count = placed;
	mesh.instanceMatrix.needsUpdate = true;
	geometry.setAttribute( 'shade', new InstancedBufferAttribute( shade, 1 ) );

	mesh.name = 'Scrub';

	return mesh;

}

// one scrub blob: an icosphere squashed into a lumpy dome, base at y = 0, with a
// baked dark-base / bright-crown gradient the material shades from
function blobGeometry() {

	const geometry = new IcosahedronGeometry( 1, 1 );
	const position = geometry.attributes.position;
	const normal = geometry.attributes.normal;
	const ao = new Float32Array( position.count );

	for ( let i = 0; i < position.count; i ++ ) {

		const ux = position.getX( i );
		const uy = position.getY( i );
		const uz = position.getZ( i );

		const h = ( uy + 1 ) / 2;
		const lump = 1 + 0.35 * Math.sin( ux * 3.1 ) * Math.sin( uy * 2.7 + 1.3 ) * Math.sin( uz * 3.5 + 2.1 );
		const r = ( 1 - 0.45 * h ) * lump;

		position.setXYZ( i, ux * r, h * 1.5, uz * r );

		// up-and-outward normals: a soft dome-lit canopy rather than faceted rock
		const inv = 1 / Math.hypot( ux, 0.6, uz );
		normal.setXYZ( i, ux * inv, 0.6 * inv, uz * inv );

		ao[ i ] = h;

	}

	geometry.setAttribute( 'ao', new BufferAttribute( ao, 1 ) );
	geometry.computeBoundingSphere();

	return geometry;

}

// the single material shared by every scrub blob and cypress: olive maquis rising
// to sunlit sage, cypress running near-black green, shaded by the baked crown gradient
function createScrubMaterial() {

	const material = new MeshStandardNodeMaterial();
	material.metalness = 0;
	material.roughness = 0.9;

	const ao = attribute( 'ao', 'float' );
	const shade = attribute( 'shade', 'float' );

	const deep = mix( color( 0x2a3a20 ), color( 0x3a4a28 ), shade );
	const bright = mix( color( 0x5a6e38 ), color( 0x7a8448 ), shade );

	// cypress ( shade = 1 ) darkens toward its blue-green column
	const lit = ao.mul( 0.55 ).add( 0.3 );
	let leaf = mix( deep, bright, lit );
	leaf = mix( leaf, mix( color( 0x1f2e1c ), color( 0x2f4426 ), ao ), smoothstep( 0.96, 1.0, shade ) );

	material.colorNode = leaf;

	return material;

}

export { VillageGenerator };
