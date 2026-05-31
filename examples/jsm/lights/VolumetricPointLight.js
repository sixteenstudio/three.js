import { PointLight, PointShadowNode, Vector3, Color, DepthTexture, HalfFloatType, RGBAFormat, LinearFilter } from 'three/webgpu';
import { int, mix, reference, texture, uniform, vec2, shadowPositionWorld } from 'three/tsl';

/**
 * Cube face look directions and up vectors ( standard OpenGL cube-map convention ), matched
 * by the sampling formula in `setupShadow`. The six faces render into the layers of a single
 * 2D array color target, sidestepping the broken cube-map color path ( and its 16-sampler
 * cost ) so point lights can carry HDR caustic refracted shadows.
 */
const _faceDirections = [
	new Vector3( 1, 0, 0 ), new Vector3( - 1, 0, 0 ), new Vector3( 0, 1, 0 ),
	new Vector3( 0, - 1, 0 ), new Vector3( 0, 0, 1 ), new Vector3( 0, 0, - 1 )
];

const _faceUps = [
	new Vector3( 0, - 1, 0 ), new Vector3( 0, - 1, 0 ), new Vector3( 0, 0, 1 ),
	new Vector3( 0, 0, - 1 ), new Vector3( 0, - 1, 0 ), new Vector3( 0, - 1, 0 )
];

const _clearColor = new Color();
const _lightPosition = new Vector3();
const _lookTarget = new Vector3();

class VolumetricPointShadowNode extends PointShadowNode {

	static get type() {

		return 'VolumetricPointShadowNode';

	}

	constructor( light ) {

		super( light );

		this.lightPosition = uniform( new Vector3() );

	}

	setupRenderTarget( shadow, builder ) {

		const size = shadow.mapSize.width;

		const depthTexture = new DepthTexture( size, size, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, 6 );
		depthTexture.name = 'VolumetricPointShadowDepth';

		const shadowMap = builder.createRenderTarget( size, size, { depth: 6, type: HalfFloatType, format: RGBAFormat, useArrayDepthTexture: true } );
		shadowMap.texture.name = 'VolumetricPointShadowMap';
		shadowMap.texture.minFilter = shadowMap.texture.magFilter = LinearFilter;
		shadowMap.depthTexture = depthTexture;

		return { shadowMap, depthTexture };

	}

	setupShadow( builder ) {

		const { shadow } = this;

		const { shadowMap } = this.setupRenderTarget( shadow, builder );

		this.shadowMap = shadowMap;
		shadow.map = shadowMap;

		const intensity = reference( 'intensity', 'float', shadow );

		// Select the cube face and its uv from the dominant axis of the light -> fragment
		// direction. Kept branchless ( `select` rather than `If` ) so the array texture is
		// sampled inline in the host material's node graph: a `setLayout` function would
		// hide the sampler binding and break the volumetric / pass compile contexts.
		const direction = shadowPositionWorld.sub( this.lightPosition );
		const absolute = direction.abs();

		const axisX = absolute.x.greaterThanEqual( absolute.y ).and( absolute.x.greaterThanEqual( absolute.z ) );
		const axisY = absolute.y.greaterThanEqual( absolute.z );

		const face = axisX.select(
			direction.x.greaterThan( 0 ).select( int( 0 ), int( 1 ) ),
			axisY.select(
				direction.y.greaterThan( 0 ).select( int( 2 ), int( 3 ) ),
				direction.z.greaterThan( 0 ).select( int( 4 ), int( 5 ) )
			)
		);

		const uv = axisX.select(
			direction.x.greaterThan( 0 )
				.select( vec2( direction.z.negate(), direction.y.negate() ), vec2( direction.z, direction.y.negate() ) )
				.div( absolute.x ),
			axisY.select(
				direction.y.greaterThan( 0 )
					.select( vec2( direction.x, direction.z ), vec2( direction.x, direction.z.negate() ) )
					.div( absolute.y ),
				direction.z.greaterThan( 0 )
					.select( vec2( direction.x, direction.y.negate() ), vec2( direction.x.negate(), direction.y.negate() ) )
					.div( absolute.z )
			)
		);

		// the render target's framebuffer maps to texture space with a flipped V, so
		// invert the vertical component to keep the projection aligned with the caster.
		const color = texture( this.shadowMap.texture, uv.mul( vec2( 0.5, - 0.5 ) ).add( 0.5 ) ).depth( face ).toVar();

		return mix( 1, color.rgb, intensity.mul( color.a ) );

	}

	renderShadow( frame ) {

		const { shadow, shadowMap, light } = this;
		const { renderer, scene } = frame;

		const camera = shadow.camera;
		camera.far = light.distance || camera.far;
		camera.updateProjectionMatrix();

		shadowMap.setSize( shadow.mapSize.width, shadow.mapSize.width, 6 );

		_lightPosition.setFromMatrixPosition( light.matrixWorld );
		this.lightPosition.value.copy( _lightPosition );

		const previousAutoClear = renderer.autoClear;
		const previousClearColor = renderer.getClearColor( _clearColor );
		const previousClearAlpha = renderer.getClearAlpha();

		renderer.autoClear = false;
		renderer.setClearColor( 0x000000, 0 );

		for ( let face = 0; face < 6; face ++ ) {

			renderer.setRenderTarget( shadowMap, face );
			renderer.clear();

			camera.position.copy( _lightPosition );
			camera.up.copy( _faceUps[ face ] );
			_lookTarget.copy( _lightPosition ).add( _faceDirections[ face ] );
			camera.lookAt( _lookTarget );
			camera.updateMatrixWorld();

			renderer.render( scene, camera );

		}

		renderer.autoClear = previousAutoClear;
		renderer.setClearColor( previousClearColor, previousClearAlpha );

	}

}

/**
 * A point light that casts HDR caustic refracted shadows by rendering its caster's
 * `castShadowNode` into six color faces of a 2D array shadow map. Drop-in for
 * `THREE.PointLight`; register the class once so the renderer resolves its light node:
 * `renderer.library.addLight( PointLightNode, VolumetricPointLight )`.
 */
class VolumetricPointLight extends PointLight {

	constructor( color, intensity, distance, decay ) {

		super( color, intensity, distance, decay );

		this.isVolumetricPointLight = true;
		this.shadow.shadowNode = new VolumetricPointShadowNode( this );

	}

}

export { VolumetricPointLight, VolumetricPointShadowNode };
