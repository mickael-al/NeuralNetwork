#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(set = 0, binding = 0) uniform UniformBufferCamShadow
{
    mat4 projview;
    vec3 pos;
    float splitDepth;
} ubS[];

layout(set = 1, binding = 0) uniform sampler2D texSampler[];

layout(set = 2, binding = 0) uniform UniformBufferModel
{
    mat4 model;
} ubo[];

layout(set = 3, binding = 0) uniform UniformBufferMaterial
{
    vec3 albedo;
    vec2 offset;
    vec2 tilling;
    float metallic;
    float roughness;    
    float normal;
    float ao;    
    uint albedoMap;
    uint normalMap;
    uint metallicMap;
    uint roughnessMap;
    uint aoMap;
    uint castShadow;
    uint orientation;
} ubm[];

layout(push_constant) uniform PushConstants
{    
    uint is;
} index;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

//Instanced
layout(location = 2) in int index_ubo;
layout(location = 3) in int index_material;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out flat int imaterial;

void main() 
{
    fragTexCoord = ubm[index_material].offset +inTexCoord * ubm[index_material].tilling;
    imaterial = index_material;

    if(ubm[index_material].orientation == 1)
    {
        vec3 camToVertex = ubS[index.is].pos - (ubo[index_ubo].model * vec4(0, 0, 0, 1)).xyz;
      
        vec3 camDirection = normalize(camToVertex);

        float angleY = -atan(camToVertex.x, camToVertex.z)+radians(90.0f);    
    
        mat3 rotationMatrixY = mat3(
            cos(angleY), 0.0, sin(angleY),
            0.0, 1.0, 0.0,
            -sin(angleY), 0.0, cos(angleY)
        );

        gl_Position = ubS[index.is].projview * ubo[index_ubo].model * vec4(rotationMatrixY * inPosition, 1.0);
    }
    else
    {
    	gl_Position = ubS[index.is].projview * ubo[index_ubo].model * vec4(inPosition, 1.0);
    }
}