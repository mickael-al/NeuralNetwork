#ifndef __SHADOW__
#define __SHADOW__

#include "Debug.hpp"
#include "VulkanMisc.hpp"
#include "BufferManager.hpp"
#include "GObject.hpp"
#include "ShadowMatrix.hpp"
#include "LightData.hpp"
#include <array>
#define TEXTURE_DIM 2048
#define SHADOW_MAP_CASCADE_COUNT 4
#define SHADOW_MAP_CUBE_COUNT 6
#define SHADOW_MAP_SPOT_COUNT 1

namespace Ge
{

	class Shadow
	{
	public:
		Shadow(VkRenderPass renderPass, LightData* light, VulkanMisc* vM);
		~Shadow();
		void updateCascades();
		std::vector<VkBuffer> getUniformBuffers();
		std::vector<VkImageView> getImageView();
		VkSampler getImageSampler() const;
		std::vector<std::vector<VkFramebuffer>> getFrameBuffer() const;
		LightData* getLightData() const;
		void createFrameBuffer(VkRenderPass renderPass);
		float aspectRatio() const;
		void updateUniformBuffer(int frame);
		void mapMemory();
	private:
		VulkanMisc* vMisc;		
		VkSampler m_textureSampler;
		std::vector<VmaBufferImage> m_depthTexture;
		std::vector<VmaBuffer> m_vmaUniformBuffers;
		std::vector<VkImageView> m_depthTextureView;
		std::vector<ShadowMatrix> m_pushConstantShadow;
		LightData* m_light;
		std::vector<std::vector<VkFramebuffer>> m_frameBuffer;

		//directionalShadowCascade
		float cascadeSplitLambda = 0.95f;
		glm::vec3 frustumCorners[8];
	};
}

#endif //!__SHADOW__