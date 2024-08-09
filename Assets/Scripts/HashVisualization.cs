using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;


public class HashVisualization : MonoBehaviour
{

    static uint RotateLeft(uint data, int steps) => (data << steps) | (data >> (32 - steps));

    public readonly struct SmallXXHash
    {
        const uint primeA = 0b10011110001101110111100110110001;
	    const uint primeB = 0b10000101111010111100101001110111;
	    const uint primeC = 0b11000010101100101010111000111101;
	    const uint primeD = 0b00100111110101001110101100101111;
	    const uint primeE = 0b00010110010101100110011110110001;

        readonly uint accumulator;

        public static implicit operator  uint(SmallXXHash hash) 
        {
            uint avalanche = hash.accumulator;

            avalanche ^= avalanche >> 15;
            avalanche *= primeB;
            avalanche ^= avalanche >> 13;
            avalanche *= primeC;
            avalanche ^= avalanche >> 16;

            return avalanche;
        }

        public SmallXXHash Eat(int data) => RotateLeft(accumulator + (uint)data * primeC, 17) * primeD;
        

        public SmallXXHash Eat(byte data) =>  RotateLeft(accumulator + data * primeE, 11) * primeA;


        public static implicit operator SmallXXHash(uint accumulator) => new SmallXXHash(accumulator);

        public static SmallXXHash seed(int seed) => (uint) seed + primeE;



        public SmallXXHash(uint accumulator)
        {
            this.accumulator = accumulator;
        }

    }


    static int
        hashesId = Shader.PropertyToID("_Hashes"),
        normalsId = Shader.PropertyToID("_Normals"),
        positionsId = Shader.PropertyToID("_Positions"),
        configId = Shader.PropertyToID("_Config");


    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS{scale = 8f};


    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(- 0.5f, 0.5f)]
    float displacement = 0.1f;

    [SerializeField]
    int seed;

    NativeArray<uint> hashes;

    NativeArray<float3> positions, normals;


    ComputeBuffer hashesBuffer;
    ComputeBuffer positionsBuffer;
    ComputeBuffer normalsBuffer;


    Bounds bounds;


    MaterialPropertyBlock propertyBlock;


    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor
    {


        [ReadOnly]
        public NativeArray<float3> positions;

        [WriteOnly]
        public NativeArray<uint> hashes;



        public SmallXXHash hash;


        public float3x4 domainTRS;

        public void Execute(int i)
        {

            /* u first to not incorporate modified v in u as it will mess up the mapping as
            we re mapping from 1D indexing to 2D grid indexing based on index = (row * colCount + column) "deducing the 'row' and 'col' from 'index'" */



            float3 p = mul(domainTRS, float4(positions[i], 1f));

            int u = (int)floor(p.x);

            int v = (int)floor(p.y);

            int w = (int)floor(p.z);

            hashes[i] = hash.Eat(u).Eat(v).Eat(w);
        }

    }

    
    bool isDirty;


    private void OnEnable() {
        
        isDirty = true;

        int length = resolution * resolution;

        hashes = new NativeArray<uint>(length, Allocator.Persistent);

        positions = new NativeArray<float3>(length, Allocator.Persistent);

        normals = new NativeArray<float3>(length, Allocator.Persistent);

        hashesBuffer = new ComputeBuffer(length, 4);

        positionsBuffer = new ComputeBuffer(length, 3 * 4);

        normalsBuffer = new ComputeBuffer(length, 3 * 4);

        propertyBlock ??= new MaterialPropertyBlock();

        propertyBlock.SetBuffer(hashesId, hashesBuffer);

        propertyBlock.SetBuffer(positionsId, positionsBuffer);

        propertyBlock.SetBuffer(normalsId, normalsBuffer);

        propertyBlock.SetVector(configId, new Vector4(resolution, 1f/ resolution, displacement));

        }

    private void OnDisable() {
        
        hashes.Dispose();
        positions.Dispose();
        normals.Dispose();
        hashesBuffer.Release();
        positionsBuffer.Release();
        normalsBuffer.Release();

        hashesBuffer = null;
        positionsBuffer = null;
        normalsBuffer = null;

    }


    private void OnValidate() {
        
        if(hashesBuffer != null && enabled)
        {
            OnDisable();
            OnEnable();
        }

    }


    private void Update() {
        
        if(isDirty || transform.hasChanged)
        {
            isDirty = false;

            transform.hasChanged = false;

            JobHandle handle = Shapes.Job.ScheduleParallel(positions, normals, resolution,transform.localToWorldMatrix, default);

            new HashJob
            {
                positions = positions,
                hashes = hashes,
                hash = SmallXXHash.seed(seed),
                domainTRS = domain.Matrix
            }.ScheduleParallel(hashes.Length, resolution, handle).Complete();


            hashesBuffer.SetData(hashes);
            positionsBuffer.SetData(positions);
            normalsBuffer.SetData(normals);


            bounds = new Bounds(transform.position, float3(2f * cmax(abs(transform.lossyScale)) + displacement));


        }


        Graphics.DrawMeshInstancedProcedural(instanceMesh, 0, material, bounds, hashes.Length, propertyBlock);
    }
}
