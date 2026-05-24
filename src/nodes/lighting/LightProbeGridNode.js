import LightingNode from './LightingNode.js';
import { NodeUpdateType } from '../core/constants.js';
import { uniform } from '../core/UniformNode.js';
import { texture3D } from '../accessors/Texture3DNode.js';
import { positionWorld } from '../accessors/Position.js';
import { normalWorld } from '../accessors/Normal.js';
import { Fn, vec3 } from '../tsl/TSLBase.js';
import { Vector3 } from '../../math/Vector3.js';

/**
 * A lighting node that adds the position-dependent diffuse irradiance of a
 * `LightProbeGrid` (a 3D grid of L2 spherical-harmonic probes) to the current
 * lighting context. It samples the grid's packed SH atlas (a single 3D texture)
 * at the fragment's world position and evaluates the irradiance for the surface
 * normal.
 *
 * This node is created automatically by {@link LightsNode} for every
 * `LightProbeGrid` in the scene, mirroring the WebGL renderer's built-in
 * `lightprobes_pars_fragment` shader chunk. It is not intended to be created
 * manually.
 *
 * @augments LightingNode
 */
class LightProbeGridNode extends LightingNode {

	static get type() {

		return 'LightProbeGridNode';

	}

	/**
	 * Constructs a new light probe grid node.
	 *
	 * @param {?Object3D} [lightProbeGrid=null] - The light probe grid (an `Object3D` with `isLightProbeGrid = true`).
	 */
	constructor( lightProbeGrid = null ) {

		super();

		/**
		 * The light probe grid.
		 *
		 * @type {?Object3D}
		 * @default null
		 */
		this.lightProbeGrid = lightProbeGrid;

		/**
		 * The minimum corner of the grid's world-space bounding box.
		 *
		 * @type {UniformNode<vec3>}
		 */
		this.probesMin = uniform( new Vector3() );

		/**
		 * The maximum corner of the grid's world-space bounding box.
		 *
		 * @type {UniformNode<vec3>}
		 */
		this.probesMax = uniform( new Vector3() );

		/**
		 * The number of probes along each axis.
		 *
		 * @type {UniformNode<vec3>}
		 */
		this.probesResolution = uniform( new Vector3() );

		/**
		 * Overwritten since light probe grid nodes are updated once per frame.
		 *
		 * @type {string}
		 * @default 'frame'
		 */
		this.updateType = NodeUpdateType.FRAME;

	}

	/**
	 * The shader is identical for every grid, so the hash only needs to mark
	 * the presence of a grid (one entry per grid reflects the grid count).
	 *
	 * @return {string} The hash.
	 */
	getHash() {

		return 'lightProbeGrid';

	}

	/**
	 * Updates the grid uniforms once per frame from the grid's bounding box and resolution.
	 *
	 * @param {NodeFrame} frame - A reference to the current node frame.
	 */
	update( /*frame*/ ) {

		const grid = this.lightProbeGrid;

		this.probesMin.value.copy( grid.boundingBox.min );
		this.probesMax.value.copy( grid.boundingBox.max );
		this.probesResolution.value.copy( grid.resolution );

	}

	setup( builder ) {

		const probesMin = this.probesMin;
		const probesMax = this.probesMax;
		const probesResolution = this.probesResolution;

		// Single atlas 3D texture that stores all 7 SH sub-volumes stacked along Z.
		// Atlas depth = 7 * ( nz + 2 ) where nz = probesResolution.z. Each sub-volume
		// occupies ( nz + 2 ) slices: 1 padding + nz data + 1 padding. Padding is a copy
		// of the first / last data slice and prevents bleeding when the hardware linear
		// filter reads across a sub-volume boundary.
		const probesSH = texture3D( this.lightProbeGrid.texture );

		const irradiance = Fn( () => {

			const res = probesResolution;
			const gridRange = probesMax.sub( probesMin );
			const resMinusOne = res.sub( 1.0 );
			const probeSpacing = gridRange.div( resMinusOne );

			// Offset sample position along the normal by half a probe spacing
			const samplePos = positionWorld.add( normalWorld.mul( probeSpacing ).mul( 0.5 ) );
			const clamped = samplePos.sub( probesMin ).div( gridRange ).clamp( 0.0, 1.0 );

			// Remap to texel centers of the probe grid ( XY and Z )
			const uvw = clamped.mul( resMinusOne ).div( res ).add( vec3( 0.5 ).div( res ) );

			// Atlas UV mapping along Z ( see lightprobes_pars_fragment.glsl.js )
			const nz = res.z;
			const paddedSlices = nz.add( 2.0 );
			const atlasDepth = paddedSlices.mul( 7.0 );
			const uvZBase = uvw.z.mul( nz ).add( 1.0 );

			const sampleAtlas = ( index ) => probesSH.sample( vec3( uvw.x, uvw.y, uvZBase.add( paddedSlices.mul( index ) ).div( atlasDepth ) ) );

			const s0 = sampleAtlas( 0.0 );
			const s1 = sampleAtlas( 1.0 );
			const s2 = sampleAtlas( 2.0 );
			const s3 = sampleAtlas( 3.0 );
			const s4 = sampleAtlas( 4.0 );
			const s5 = sampleAtlas( 5.0 );
			const s6 = sampleAtlas( 6.0 );

			// Unpack 9 vec3 SH L2 coefficients
			const c0 = s0.xyz;
			const c1 = vec3( s0.w, s1.x, s1.y );
			const c2 = vec3( s1.z, s1.w, s2.x );
			const c3 = vec3( s2.y, s2.z, s2.w );
			const c4 = s3.xyz;
			const c5 = vec3( s3.w, s4.x, s4.y );
			const c6 = vec3( s4.z, s4.w, s5.x );
			const c7 = vec3( s5.y, s5.z, s5.w );
			const c8 = s6.xyz;

			// Evaluate L2 irradiance
			const x = normalWorld.x, y = normalWorld.y, z = normalWorld.z;

			let result = c0.mul( 0.886227 );
			result = result.add( c1.mul( 2.0 * 0.511664 ).mul( y ) );
			result = result.add( c2.mul( 2.0 * 0.511664 ).mul( z ) );
			result = result.add( c3.mul( 2.0 * 0.511664 ).mul( x ) );
			result = result.add( c4.mul( 2.0 * 0.429043 ).mul( x ).mul( y ) );
			result = result.add( c5.mul( 2.0 * 0.429043 ).mul( y ).mul( z ) );
			result = result.add( c6.mul( z.mul( z ).mul( 0.743125 ).sub( 0.247708 ) ) );
			result = result.add( c7.mul( 2.0 * 0.429043 ).mul( x ).mul( z ) );
			result = result.add( c8.mul( 0.429043 ).mul( x.mul( x ).sub( y.mul( y ) ) ) );

			return result.max( 0.0 );

		} )();

		builder.context.irradiance.addAssign( irradiance );

	}

}

export default LightProbeGridNode;
