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


layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in flat int imaterial;

void main() 
{
	vec4 cr = texture(texSampler[ubm[imaterial].albedoMap], fragTexCoord).rgba;
    if(cr.a == 0.0 || ubm[imaterial].castShadow == 1 || ubm[imaterial].castShadow == 3)
    {
        discard;
        return;
    }
}