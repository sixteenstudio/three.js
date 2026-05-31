import { PointLight, PointShadowNode, Vector3, Color, DepthTexture, HalfFloatType, RGBAFormat, LinearFilter } from 'three/webgpu';
import { Fn, If, int, mix, reference, texture, uniform, vec2, shadowPositionWorld } from 'three/tsl';

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

		const sampleCaustic = Fn( ( [ direction ] ) => {

			const absolute = direction.abs().toVar();
			const face = int( 0 ).toVar();
			const uv = vec2( 0 ).toVar();

			If( absolute.x.greaterThanEqual( absolute.y ).and( absolute.x.greaterThanEqual( absolute.z ) ), () => {

				If( direction.x.greaterThan( 0 ), () => { face.assign( 0 ); uv.assign( vec2( direction.z.negate(), direction.y.negate() ).div( absolute.x ) ); } )
					.Else( () => { face.assign( 1 ); uv.assign( vec2( direction.z, direction.y.negate() ).div( absolute.x ) ); } );

			} ).ElseIf( absolute.y.greaterThanEqual( absolute.z ), () => {

				If( direction.y.greaterThan( 0 ), () => { face.assign( 2 ); uv.assign( vec2( direction.x, direction.z ).div( absolute.y ) ); } )
					.Else( () => { face.assign( 3 ); uv.assign( vec2( direction.x, direction.z.negate() ).div( absolute.y ) ); } );

			} ).Else( () => {

				If( direction.z.greaterThan( 0 ), () => { face.assign( 4 ); uv.assign( vec2( direction.x, direction.y.negate() ).div( absolute.z ) ); } )
					.Else( () => { face.assign( 5 ); uv.assign( vec2( direction.x.negate(), direction.y.negate() ).div( absolute.z ) ); } );

			} );

			return texture( this.shadowMap.texture, uv.mul( 0.5 ).add( 0.5 ) ).depth( face );

		} ).setLayout( { name: 'volumetricPointCaustic_' + this.light.id, type: 'vec4', inputs: [ { name: 'direction', type: 'vec3' } ] } );

		const color = sampleCaustic( shadowPositionWorld.sub( this.lightPosition ) ).toVar();

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
