using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using static MathExtensions;


public class HashVisualization : Visualization
{

    static int hashesId = Shader.PropertyToID("_Hashes");


    [SerializeField]
    SpaceTRS domain = new SpaceTRS{scale = 8f};


    [SerializeField]
    int seed;


    NativeArray<uint4> hashes;

    ComputeBuffer hashesBuffer;


    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor
    {


        [ReadOnly]
        public NativeArray<float3x4> positions;

        [WriteOnly]
        public NativeArray<uint4> hashes;



        public SmallXXHash4 hash;


        public float3x4 domainTRS;


        public void Execute(int i)
        {

            /* u first to not incorporate modified v in u as it will mess up the mapping as
            we re mapping from 1D indexing to 2D grid indexing based on index = (row * colCount + column) "deducing the 'row' and 'col' from 'index'" */

            float4x3 p = domainTRS.TransformVectors(transpose(positions[i]));

            int4 u = (int4)floor(p.c0);

            int4 v = (int4)floor(p.c1);

            int4 w = (int4)floor(p.c2);

            hashes[i] = hash.Eat(u).Eat(v).Eat(w);
        }

    }

    

    protected override void EnableVisualization(int dataLength, MaterialPropertyBlock propertyBlock) {
        
        hashes = new NativeArray<uint4>(dataLength, Allocator.Persistent);

        // positions = new NativeArray<float3x4>(length, Allocator.Persistent);

        // normals = new NativeArray<float3x4>(length, Allocator.Persistent);

        hashesBuffer = new ComputeBuffer(dataLength * 4, 4);

        propertyBlock.SetBuffer(hashesId, hashesBuffer);

    }

    protected override void DisableVisualization() {
        
        hashes.Dispose();
        hashesBuffer.Release();
        hashesBuffer = null;
    }


    protected override void UpdateVisualization(NativeArray<float3x4> positions, int resolution, JobHandle handle)
    {
        
        new HashJob
        {
            positions = positions,
            hashes = hashes,
            hash = SmallXXHash.Seed(seed),
            domainTRS = domain.Matrix
        }.ScheduleParallel(hashes.Length, resolution, handle).Complete();

        hashesBuffer.SetData(hashes.Reinterpret<uint>( 4 * 4));
    }
}
