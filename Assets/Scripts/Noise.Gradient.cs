using Unity.Mathematics;
using static Unity.Mathematics.math;
using static Visualization;
public static partial class Noise
{
    public interface IGradient{
        float4 Evaluate (SmallXXHash4 hash, float4 x);

		float4 Evaluate (SmallXXHash4 hash, float4 x, float4 y);

		float4 Evaluate (SmallXXHash4 hash, float4 x, float4 y, float4 z);
    }

    public struct Value : IGradient
    {
        
        public float4 Evaluate (SmallXXHash4 hash, float4 x) => hash.Floats01A * 2f - 1f;

		public float4 Evaluate (SmallXXHash4 hash, float4 x, float4 y) => hash.Floats01A * 2f - 1f;

		public float4 Evaluate (SmallXXHash4 hash, float4 x, float4 y, float4 z) => hash.Floats01A * 2f - 1f;
    }


}
