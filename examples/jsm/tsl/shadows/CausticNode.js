import { Fn, refract, div, positionViewDirection, normalView, float, vec2, vec3 } from 'three/tsl';

/**
 * @module CausticNode
 * @three_import import { caustic } from 'three/addons/tsl/shadows/CausticNode.js';
 */

/**
 * Computes a refracted caustic projection for a transparent caster, intended to be
 * assigned to a material's {@link NodeMaterial#castShadowNode} together with
 * `renderer.shadowMap.transmitted = true`. Evaluated from the light's point of view
 * during the shadow pass, it bends the incoming direction through the surface normal
 * and projects the resulting vector onto a tiling caustic texture, sampling each
 * channel with a small offset to fake wavelength-dependent dispersion. Grazing angles
 * are faded out so the caustic concentrates where the surface faces the light. The
 * returned color is left white so each light tints its own caustic automatically.
 *
 * @tsl
 * @function
 * @param {Node<vec4>} causticMap - The tiling caustic texture node to sample, e.g. `texture( map )`.
 * @param {Node<float>} [ior=1.5] - The index of refraction used to bend the incoming direction.
 * @param {Node<float>} [occlusion=12] - The grazing-angle falloff power.
 * @param {Node<float>} [scale=0.6] - The caustic texture projection scale.
 * @param {Node<float>} [dispersion=0.004] - The per-channel offset faking chromatic dispersion.
 * @param {Node<float>} [intensity=25] - The brightness gain of the focused caustic ( use an HDR shadow map ).
 * @return {Node<vec3>} The caustic projection color.
 *
 * @example
 * material.castShadowNode = vec4( caustic( texture( causticMap ), material.ior ), 0.85 );
 */
export const caustic = Fn( ( [ causticMap, ior = float( 1.5 ), occlusion = float( 12 ), scale = float( 0.6 ), dispersion = float( 0.004 ), intensity = float( 25 ) ] ) => {

	const refraction = refract( positionViewDirection.negate(), normalView, div( 1.0, ior ) ).normalize();
	const facing = normalView.z.pow( occlusion );

	const causticUV = refraction.xy.mul( scale );
	const offset = normalView.z.pow( - 0.9 ).mul( dispersion );

	const projection = vec3(
		causticMap.sample( causticUV.add( vec2( offset.negate(), 0 ) ) ).r,
		causticMap.sample( causticUV.add( vec2( 0, offset.negate() ) ) ).g,
		causticMap.sample( causticUV.add( vec2( offset, offset ) ) ).b
	);

	// focused caustic ( HDR ) over a dim refracted-transmission floor
	return projection.mul( facing.mul( intensity ) ).add( facing );

} );
