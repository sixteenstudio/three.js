import * as THREE from 'three';
import {
	Fn, abs, add, color, dot, float, length, max, min, mix, mx_noise_float, mx_noise_vec3,
	normalLocal, normalView, positionLocal, positionWorld, pow, saturate, sin, smoothstep,
	step, time, vec2, vec3, vec4, normalize
} from 'three/tsl';

const SUITS = {
	emerald: { base: '#0f7a4f', deep: '#063a24', highlight: '#6dffb8', ior: 1.57 },
	ruby: { base: '#b01028', deep: '#4a0610', highlight: '#ff8a9a', ior: 1.77 },
	sapphire: { base: '#1a3fc8', deep: '#08164f', highlight: '#9ec4ff', ior: 1.77 },
	amethyst: { base: '#6b2f96', deep: '#2a103f', highlight: '#d7a8ff', ior: 1.55 }
};

const faceCoords = Fn( () => {

	const n = abs( normalLocal );
	const dominant = max( n.x, max( n.y, n.z ) );
	const u = mix( positionLocal.z, positionLocal.x, step( n.y, n.x ).mul( step( n.z, n.x ) ) );
	const v = mix( positionLocal.y, mix( positionLocal.z, positionLocal.y, step( n.x, n.z ) ), step( n.x, n.y ).mul( step( n.z, n.y ) ) );
	const mask = step( dominant.sub( 0.01 ), dominant );

	return vec2( u, v ).mul( mask );

} );

const faceBorderMask = Fn( ( [ width = float( 0.045 ), softness = float( 0.012 ) ] ) => {

	const uv = faceCoords();
	const edge = min( abs( uv.x ), abs( uv.y ) );
	const inner = float( 0.5 ).sub( width );
	return smoothstep( inner.add( softness ), inner.sub( softness ), edge );

} );

const etchedCarve = Fn( ( [ baseColor, baseRoughness ] ) => {

	const border = faceBorderMask( 0.05, 0.015 );
	const groove = faceBorderMask( 0.035, 0.008 );
	const carved = baseColor.mul( mix( float( 0.72 ), float( 1.0 ), border ) );
	const rough = mix( baseRoughness, baseRoughness.add( 0.18 ), groove );
	return vec4( carved, rough );

} );

const auroraBands = Fn( ( [ t ] ) => {

	const p = positionWorld.mul( 0.8 );
	const wave = sin( p.x.mul( 1.7 ).add( t.mul( 0.45 ) ) )
		.add( cos( p.z.mul( 2.1 ).sub( t.mul( 0.35 ) ) ) )
		.add( mx_noise_float( p.mul( 1.3 ).add( t.mul( 0.08 ) ) ).mul( 1.4 ) );
	const band = smoothstep( 0.15, 0.85, wave.mul( 0.5 ).add( 0.5 ) );
	const tint = mix( color( '#0a4f55' ), color( '#7ef7d0' ), band );
	return tint.mul( band.mul( 0.55 ).add( 0.12 ) );

} );

const neonRim = Fn( ( [ t, tint = color( '#58ffb0' ) ] ) => {

	const rim = pow( float( 1.0 ).sub( saturate( dot( normalView, vec3( 0, 0, 1 ) ) ) ), 2.2 );
	const pulse = sin( t.mul( 2.4 ) ).mul( 0.15 ).add( 0.85 );
	return tint.mul( rim.mul( pulse ).mul( 1.4 ) );

} );

const iridescentColor = Fn( ( [ t, viewNormal ] ) => {

	const n = normalize( viewNormal.add( vec3( sin( t.mul( 0.7 ) ), cos( t.mul( 0.5 ) ), 0.0 ).mul( 0.15 ) ) );
	const angle = dot( n, normalize( vec3( 1, 1, 0.4 ) ) ).mul( 6.0 ).add( t.mul( 0.8 ) );
	const r = sin( angle ).mul( 0.5 ).add( 0.5 );
	const g = sin( angle.add( 2.1 ) ).mul( 0.5 ).add( 0.5 );
	const b = sin( angle.add( 4.2 ) ).mul( 0.5 ).add( 0.5 );
	return vec3( r, g, b ).pow( vec3( 1.2 ) );

} );

const gemInterior = Fn( ( [ suitColors, sparkleScale = float( 18.0 ) ] ) => {

	const p = positionLocal.mul( sparkleScale );
	const n1 = mx_noise_float( p );
	const n2 = mx_noise_float( p.mul( 2.7 ).add( 17.3 ) );
	const n3 = mx_noise_vec3( p.mul( 1.4 ) );
	const depth = mix( suitColors.deep, suitColors.base, n1.mul( 0.65 ).add( 0.35 ) );
	const inclusion = smoothstep( 0.82, 0.95, n2 ).mul( 0.25 );
	const sparkle = smoothstep( 0.92, 0.995, mx_noise_float( p.mul( 6.0 ).add( n3.xy ) ) );
	const facet = abs( sin( p.x.mul( 3.1 ) ).mul( sin( p.y.mul( 2.7 ) ) ).mul( sin( p.z.mul( 3.5 ) ) ) );
	const gem = mix( depth, suitColors.highlight, facet.mul( 0.18 ).add( sparkle.mul( 0.45 ) ) );
	return gem.sub( inclusion );

} );

const ceramicBody = Fn( () => {

	const grain = mx_noise_float( positionLocal.mul( 18.0 ) ).mul( 0.04 );
	const edgeWear = pow( float( 1.0 ).sub( saturate( dot( normalView, vec3( 0, 0, 1 ) ) ) ), 3.0 ).mul( 0.06 );
	const base = color( '#f4f1ea' ).mul( float( 1.0 ).sub( grain ).sub( edgeWear ) );
	return vec4( base, float( 0.28 ) );

} );

const chalkBody = Fn( () => {

	const pores = mx_noise_float( positionLocal.mul( 42.0 ) );
	const dust = mx_noise_float( positionLocal.mul( 9.0 ) ).mul( 0.08 );
	const edge = pow( float( 1.0 ).sub( saturate( dot( normalView, vec3( 0, 0, 1 ) ) ) ), 2.0 ).mul( 0.12 );
	const base = color( '#ddd5c8' ).mul( float( 0.96 ).sub( dust ) );
	const rough = float( 0.92 ).add( pores.mul( 0.06 ) ).add( edge );
	return vec4( base, rough );

} );

const ivoryBody = Fn( () => {

	const grain = mx_noise_float( positionLocal.mul( vec3( 28.0, 4.0, 1.0 ) ) );
	const warm = mix( color( '#f0dcc0' ), color( '#fff1df' ), grain );
	const age = mx_noise_float( positionLocal.mul( 6.0 ) ).mul( 0.07 );
	const body = warm.mul( float( 1.0 ).sub( age ) );
	return vec4( body, float( 0.42 ) );

} );

const crystalBody = Fn( ( [ t ] ) => {

	const cloud = mx_noise_vec3( positionLocal.mul( 4.0 ).add( t.mul( 0.02 ) ) ).mul( 0.08 );
	const caustic = pow( abs( sin( positionLocal.x.mul( 12.0 ).add( positionLocal.y.mul( 9.0 ) ) ) ), 6.0 ).mul( 0.25 );
	const tint = color( '#eef6ff' ).add( cloud ).add( caustic );
	return vec4( tint, float( 0.04 ) );

} );

const obsidianBody = Fn( () => {

	const smoke = mx_noise_float( positionLocal.mul( 3.5 ) ).mul( 0.08 );
	const gloss = pow( saturate( dot( normalView, normalize( vec3( 0.4, 0.8, 0.5 ) ) ) ), 8.0 ).mul( 0.35 );
	const base = color( '#070708' ).add( smoke ).add( gloss );
	return vec4( base, float( 0.08 ) );

} );

const leadBody = Fn( () => {

	const oxide = mx_noise_float( positionLocal.mul( 11.0 ) ).mul( 0.06 );
	const scuff = mx_noise_float( positionLocal.mul( 35.0 ) ).mul( 0.03 );
	const base = color( '#6f7378' ).mul( float( 1.0 ).sub( oxide ).sub( scuff ) );
	return vec4( base, float( 0.78 ) );

} );

const gildedBody = Fn( () => {

	const scratch = mx_noise_float( positionLocal.mul( 90.0 ) ).mul( 0.08 );
	const wear = pow( float( 1.0 ).sub( saturate( dot( normalView, vec3( 0, 0, 1 ) ) ) ), 2.5 ).mul( 0.18 );
	const gold = mix( color( '#c89a2b' ), color( '#ffe7a3' ), float( 0.35 ).sub( scratch ) );
	const body = gold.mul( float( 1.0 ).sub( wear ) );
	return vec4( body, float( 0.18 ) );

} );

const loadedBody = Fn( ( [ t ] ) => {

	const ceramic = ceramicBody();
	const coreHint = smoothstep( 0.15, 0.0, length( positionLocal.sub( vec3( 0.08, - 0.06, 0.05 ) ) ) );
	const asym = mx_noise_float( positionLocal.mul( 5.0 ).add( t.mul( 0.01 ) ) ).mul( 0.04 );
	const tint = ceramic.xyz.sub( coreHint.mul( 0.08 ) ).sub( asym );
	return vec4( tint, ceramic.w );

} );

const prismBody = Fn( ( [ t ] ) => {

	const iri = iridescentColor( t, normalLocal );
	const glass = mix( color( '#ffffff' ), iri, float( 0.35 ) );
	return vec4( glass, float( 0.02 ) );

} );

const BODY_BUILDERS = {
	ceramic: ceramicBody,
	chalk: chalkBody,
	ivory: ivoryBody,
	crystal: () => crystalBody( time ),
	obsidian: obsidianBody,
	lead: leadBody,
	gilded: gildedBody,
	loaded: () => loadedBody( time ),
	prism: () => prismBody( time )
};

function applyFinishToBody( bodyColor, bodyRoughness, bodyMetalness, finish, t ) {

	let colorNode = bodyColor;
	let roughnessNode = bodyRoughness;
	let metalnessNode = bodyMetalness;
	let emissiveNode = vec3( 0 );

	if ( finish === 'etched' ) {

		const etched = etchedCarve( bodyColor, bodyRoughness );
		colorNode = etched.xyz;
		roughnessNode = etched.w;

	} else if ( finish === 'neon' ) {

		emissiveNode = add( emissiveNode, neonRim( t ) );
		colorNode = mix( bodyColor, color( '#d8ffe9' ), float( 0.12 ) );
		roughnessNode = bodyRoughness.mul( 0.75 );

	} else if ( finish === 'aurora' ) {

		const aurora = auroraBands( t );
		emissiveNode = add( emissiveNode, aurora );
		colorNode = mix( bodyColor, aurora, float( 0.22 ) );

	}

	return { colorNode, roughnessNode, metalnessNode, emissiveNode };

}

function createGemPipMaterial( suit, finish, tNode ) {

	const suitColors = {
		base: color( SUITS[ suit ].base ),
		deep: color( SUITS[ suit ].deep ),
		highlight: color( SUITS[ suit ].highlight )
	};

	const material = new THREE.MeshPhysicalNodeMaterial();
	const gemColor = gemInterior( suitColors );

	material.colorNode = gemColor;
	material.roughnessNode = float( 0.06 ).add( mx_noise_float( positionLocal.mul( 20.0 ) ).mul( 0.04 ) );
	material.metalnessNode = float( 0.0 );
	material.iorNode = float( SUITS[ suit ].ior );
	material.transmissionNode = float( 0.35 );
	material.thicknessNode = float( 0.65 );
	material.clearcoatNode = float( 1.0 );
	material.clearcoatRoughnessNode = float( 0.04 );
	material.emissiveNode = smoothstep( 0.93, 0.995, mx_noise_float( positionLocal.mul( 40.0 ).add( tNode.mul( 0.2 ) ) ) ).mul( suitColors.highlight ).mul( 0.35 );

	if ( finish === 'neon' ) {

		material.emissiveNode = add( material.emissiveNode, neonRim( tNode, color( '#7dffb8' ) ).mul( 1.5 ) );

	} else if ( finish === 'aurora' ) {

		material.emissiveNode = add( material.emissiveNode, auroraBands( tNode ).mul( 0.35 ) );

	}

	return material;

}

function createPrismPipMaterial( finish, tNode ) {

	const material = new THREE.MeshPhysicalNodeMaterial();
	const iri = iridescentColor( tNode, normalLocal );

	material.colorNode = iri;
	material.roughnessNode = float( 0.03 );
	material.metalnessNode = float( 0.15 );
	material.iorNode = float( 2.0 );
	material.transmissionNode = float( 0.25 );
	material.clearcoatNode = float( 1.0 );
	material.emissiveNode = iri.mul( 0.08 );

	if ( finish === 'neon' ) {

		material.emissiveNode = add( material.emissiveNode, neonRim( tNode, iri ).mul( 1.2 ) );

	}

	return material;

}

function createLoadedPlugMaterial() {

	const material = new THREE.MeshPhysicalNodeMaterial();
	const grain = mx_noise_float( positionLocal.mul( 25.0 ) ).mul( 0.08 );

	material.colorNode = color( '#3a3f45' ).mul( float( 1.0 ).sub( grain ) );
	material.roughnessNode = float( 0.35 ).add( grain );
	material.metalnessNode = float( 0.85 );
	material.emissiveNode = vec3( 0 );

	return material;

}

function createBodyMaterial( bodyType, finish, tNode ) {

	const builder = BODY_BUILDERS[ bodyType ] || ceramicBody;
	const sample = builder();
	const baseMetalness = bodyType === 'gilded' ? float( 1.0 ) : ( bodyType === 'lead' ? float( 0.55 ) : float( 0.0 ) );

	let {
		colorNode,
		roughnessNode,
		metalnessNode,
		emissiveNode
	} = applyFinishToBody( sample.xyz, sample.w, baseMetalness, finish, tNode );

	const material = new THREE.MeshPhysicalNodeMaterial();
	material.colorNode = colorNode;
	material.roughnessNode = roughnessNode;
	material.metalnessNode = metalnessNode;
	material.emissiveNode = emissiveNode;

	if ( bodyType === 'crystal' || bodyType === 'prism' ) {

		material.transmissionNode = bodyType === 'prism' ? float( 0.92 ) : float( 0.78 );
		material.thicknessNode = float( 0.85 );
		material.iorNode = bodyType === 'prism' ? float( 2.2 ) : float( 1.48 );
		material.transparent = true;

	}

	if ( bodyType === 'ivory' ) {

		material.transmissionNode = float( 0.08 );
		material.thicknessNode = float( 0.35 );

	}

	if ( bodyType === 'obsidian' ) {

		material.clearcoatNode = float( 1.0 );
		material.clearcoatRoughnessNode = float( 0.03 );

	}

	if ( bodyType === 'gilded' ) {

		material.clearcoatNode = float( 0.35 );
		material.clearcoatRoughnessNode = float( 0.18 );

	}

	if ( bodyType === 'loaded' ) {

		material.transmissionNode = float( 0.04 );
		material.thicknessNode = float( 0.2 );

	}

	return material;

}

/**
 * Build body and pip materials for a collectible die configuration.
 *
 * @param {Object} config
 * @param {string} [config.suit='emerald']
 * @param {string} [config.body='ceramic']
 * @param {string|null} [config.finish=null]
 * @param {Node} [config.timeNode=time]
 * @returns {{ body: MeshPhysicalNodeMaterial, pip: MeshPhysicalNodeMaterial|null, plug: MeshPhysicalNodeMaterial|null }}
 */
function createDiceMaterials( config = {} ) {

	const suit = config.suit !== undefined ? config.suit : 'emerald';
	const body = config.body !== undefined ? config.body : 'ceramic';
	const finish = config.finish !== undefined ? config.finish : null;
	const tNode = config.timeNode !== undefined ? config.timeNode : time;

	if ( body === 'lead' ) {

		return {
			body: createBodyMaterial( 'lead', finish, tNode ),
			pip: null,
			plug: null
		};

	}

	if ( body === 'prism' ) {

		return {
			body: createBodyMaterial( 'prism', finish, tNode ),
			pip: createPrismPipMaterial( finish, tNode ),
			plug: null
		};

	}

	const pip = createGemPipMaterial( suit, finish, tNode );
	const plug = body === 'loaded' ? createLoadedPlugMaterial() : null;

	return {
		body: createBodyMaterial( body, finish, tNode ),
		pip,
		plug
	};

}

export {
	SUITS,
	createDiceMaterials,
	createBodyMaterial,
	createGemPipMaterial,
	createPrismPipMaterial,
	createLoadedPlugMaterial
};
