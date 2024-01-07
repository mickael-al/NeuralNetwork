#include "Shadow.hpp"
#include "ImageViewSwapChains.hpp"
#include "CameraManager.hpp"

namespace Ge
{
	Shadow::Shadow(VkRenderPass renderPass, LightData* light, VulkanMisc* vM)
	{
		vMisc = vM;
		m_light = light;
		int imageBuffer = m_light->ubl->status == 1 ? SHADOW_MAP_CUBE_COUNT : m_light->ubl->status == 2 ? SHADOW_MAP_SPOT_COUNT : SHADOW_MAP_CASCADE_COUNT;
		VkFormat depthFormat = vM->str_VulkanSwapChainMisc->str_depthFormat;
		m_depthTexture.resize(imageBuffer);
		for (int i = 0; i < imageBuffer; i++)
		{
			VmaBuffer buffer;
			if (!BufferManager::createBuffer(sizeof(ShadowMatrix), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, vM->str_VulkanDeviceMisc))
			{
				Debug::Error("Echec de la creation d'un uniform buffer");
			}
			m_vmaUniformBuffers.push_back(buffer);			

			BufferManager::createImageBuffer(TEXTURE_DIM, TEXTURE_DIM, VK_IMAGE_TYPE_2D, 1, 1, VK_SAMPLE_COUNT_1_BIT, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthTexture[i], 0, vM);

			VkImageSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1;
			subresourceRange.layerCount = 1;

			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = m_depthTexture[i].image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = depthFormat;
			viewInfo.subresourceRange = subresourceRange;
			
			VkImageView imageView;
			if (vkCreateImageView(vM->str_VulkanDeviceMisc->str_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
			{
				Debug::Error("Echec de la creation d'une image vue");
			}

			m_depthTextureView.push_back(imageView);
		}

		for (int i = 0; i < m_depthTextureView.size(); i++)
		{
			ShadowMatrix pc_empty;
			pc_empty.splitDepth = 0.0f;
			pc_empty.projview = glm::mat4();
			m_pushConstantShadow.push_back(pc_empty);
		}
		createFrameBuffer(renderPass);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		if (vkCreateSampler(vM->str_VulkanDeviceMisc->str_device, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
		{
			Debug::Error("Echec de la creation d'un sampler de texture");
		}
		mapMemory();
	}

	void Shadow::updateCascades()
	{
		Camera* currentCamera = CameraManager::getCameraManager()->getCurrentCamera();
		float cascadeSplits[SHADOW_MAP_CASCADE_COUNT];

		float nearClip = currentCamera->getNear();
		float farClip = currentCamera->getFar();
		float clipRange = farClip - nearClip;

		float minZ = nearClip;
		float maxZ = nearClip + clipRange;

		float range = maxZ - minZ;
		float ratio = maxZ / minZ;

		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++)
		{
			float p = (i + 1) / static_cast<float>(SHADOW_MAP_CASCADE_COUNT);
			float log = minZ * std::pow(ratio, p);
			float uniform = minZ + range * p;
			float d = cascadeSplitLambda * (log - uniform) + uniform;
			cascadeSplits[i] = (d - nearClip) / clipRange;
		}

		// Calculate orthographic projection matrix for each cascade
		float lastSplitDist = 0.0;
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++)
		{
			float splitDist = cascadeSplits[i];

			frustumCorners[0] = glm::vec3(-1.0f, 1.0f, 0.0f);
			frustumCorners[1] = glm::vec3(1.0f, 1.0f, 0.0f);
			frustumCorners[2] = glm::vec3(1.0f, -1.0f, 0.0f);
			frustumCorners[3] = glm::vec3(-1.0f, -1.0f, 0.0f);
			frustumCorners[4] = glm::vec3(-1.0f, 1.0f, 1.0f);
			frustumCorners[5] = glm::vec3(1.0f, 1.0f, 1.0f);
			frustumCorners[6] = glm::vec3(1.0f, -1.0f, 1.0f);
			frustumCorners[7] = glm::vec3(-1.0f, -1.0f, 1.0f);			
			
			glm::mat4 invCam = glm::inverse(currentCamera->getProjectionMatrix() * currentCamera->getViewMatrix());			
			for (uint32_t i = 0; i < 8; i++)
			{
				glm::vec4 invCorner = invCam * glm::vec4(frustumCorners[i], 1.0f);
				frustumCorners[i] = invCorner / invCorner.w;
			}

			for (uint32_t i = 0; i < 4; i++)
			{
				glm::vec3 dist = frustumCorners[i + 4] - frustumCorners[i];
				frustumCorners[i + 4] = frustumCorners[i] + (dist * splitDist);
				frustumCorners[i] = frustumCorners[i] + (dist * lastSplitDist);
			}

			// Get frustum center
			glm::vec3 frustumCenter = glm::vec3(0.0f);
			for (uint32_t i = 0; i < 8; i++)
			{
				frustumCenter += frustumCorners[i];
			}
			frustumCenter /= 8.0f;

			float radius = 0.0f;
			for (uint32_t i = 0; i < 8; i++)
			{
				float distance = glm::length(frustumCorners[i] - frustumCenter);
				radius = glm::max(radius, distance);
			}
			radius = std::ceil(radius * 16.0f) / 16.0f;

			glm::vec3 maxExtents = glm::vec3(radius);
			glm::vec3 minExtents = -maxExtents;

			m_pushConstantShadow[i].pos = frustumCenter - glm::normalize(m_light->ubl->direction) * -minExtents.z;
			glm::mat4 lightViewMatrix = glm::lookAt(m_pushConstantShadow[i].pos, frustumCenter, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 lightOrthoMatrix = glm::ortho(minExtents.x, maxExtents.x, maxExtents.y, minExtents.y, 0.0f, maxExtents.z - minExtents.z);

			m_pushConstantShadow[i].splitDepth = (nearClip + splitDist * clipRange) * -1.0f;
			m_pushConstantShadow[i].projview = lightOrthoMatrix * lightViewMatrix;
			lastSplitDist = cascadeSplits[i];
		}
	}

	std::vector<VkImageView> Shadow::getImageView()
	{
		return m_depthTextureView;
	}

	std::vector<VkBuffer> Shadow::getUniformBuffers()
	{
		std::vector<VkBuffer> buffers;
		for (int i = 0; i < m_vmaUniformBuffers.size(); i++)
		{
			buffers.push_back(m_vmaUniformBuffers[i].buffer);
		}
		return buffers;
	}

	VkSampler Shadow::getImageSampler() const
	{
		return m_textureSampler;
	}

	std::vector<std::vector<VkFramebuffer>> Shadow::getFrameBuffer() const
	{
		return m_frameBuffer;
	}

	LightData* Shadow::getLightData() const
	{
		return m_light;
	}

	void Shadow::createFrameBuffer(VkRenderPass renderPass)
	{
		for (int i = 0; i < m_frameBuffer.size(); i++)
		{
			for (int j = 0; j < m_frameBuffer[i].size(); j++)
			{
				vkDestroyFramebuffer(vMisc->str_VulkanDeviceMisc->str_device, m_frameBuffer[i][j], nullptr);
			}
		}
		m_frameBuffer.clear();
		std::vector<VkImageView> swapChainImageViews = vMisc->str_VulkanSwapChainMisc->str_swapChainImageViews;
		VkImageView attachments[1];
		VkFramebuffer frameBuffer;
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;		
		framebufferInfo.width = TEXTURE_DIM;
		framebufferInfo.height = TEXTURE_DIM;
		framebufferInfo.layers = 1;		

		for (int j = 0; j < m_depthTextureView.size(); j++)
		{
			attachments[0] = m_depthTextureView[j];			
			std::vector<VkFramebuffer> frame;
			for (size_t i = 0; i < swapChainImageViews.size(); i++)
			{
				if (vkCreateFramebuffer(vMisc->str_VulkanDeviceMisc->str_device, &framebufferInfo, nullptr, &frameBuffer) != VK_SUCCESS)
				{
					Debug::Error("Echec de la creation d'un framebuffer");
				}
				frame.push_back(frameBuffer);
			}
			m_frameBuffer.push_back(frame);
		}
	}

	float Shadow::aspectRatio() const
	{
		return 1.0f;
	}

	void Shadow::mapMemory()
	{
		if (m_light->ubl->status == 0)
		{
			updateCascades();
			for (int i = 0; i < m_pushConstantShadow.size(); i++)
			{
				memcpy(BufferManager::mapMemory(m_vmaUniformBuffers[i]), &m_pushConstantShadow[i], sizeof(ShadowMatrix));
				BufferManager::unMapMemory(m_vmaUniformBuffers[i]);
			}
		}
		else if (m_light->ubl->status == 1)
		{
			for (int i = 0; i < m_pushConstantShadow.size(); i++)
			{
				updateUniformBuffer(i);
				memcpy(BufferManager::mapMemory(m_vmaUniformBuffers[i]), &m_pushConstantShadow[i], sizeof(ShadowMatrix));
				BufferManager::unMapMemory(m_vmaUniformBuffers[i]);
			}
		}
	}

	void Shadow::updateUniformBuffer(int frame)
	{
		m_pushConstantShadow[frame].pos = m_light->transform->position;
		m_pushConstantShadow[frame].projview = glm::inverse(glm::translate(glm::mat4(1.0f), m_pushConstantShadow[frame].pos) * glm::toMat4(glm::quat(m_light->transform->rotation)) * glm::scale(glm::mat4(1.0f), m_light->transform->scale));
		glm::mat4 projectionMatrix;
		bool m_ortho = false;
		float m_orthoSize = 10.0f;
		float m_near = 0.01f;
		float m_far = 1000.0f;
		float m_fov = m_light->ubl->spotAngle;
		if (m_ortho)
		{
			float halfHeight = m_orthoSize * 0.5f;
			float halfWidth = halfHeight * aspectRatio();
			projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, m_near, m_far);
		}
		else
		{
			projectionMatrix = glm::perspective(glm::radians(m_fov), aspectRatio(), m_near, m_far);
		}

		m_pushConstantShadow[frame].projview = glm::scale(projectionMatrix, glm::vec3(1.0f, -1.0f, 1.0f));
	}

	Shadow::~Shadow()
	{
		for (int i = 0; i < m_frameBuffer.size(); i++)
		{
			for (int j = 0; j < m_frameBuffer[i].size(); j++)
			{
				vkDestroyFramebuffer(vMisc->str_VulkanDeviceMisc->str_device, m_frameBuffer[i][j], nullptr);
			}
		}
		m_frameBuffer.clear();
		for (int i = 0; i < m_depthTextureView.size(); i++)
		{
			vkDestroyImageView(vMisc->str_VulkanDeviceMisc->str_device, m_depthTextureView[i], nullptr);
		}
		vkDestroySampler(vMisc->str_VulkanDeviceMisc->str_device, m_textureSampler, nullptr);
		m_depthTextureView.clear();
		for (int i = 0; i < m_depthTexture.size(); i++)
		{
			BufferManager::destroyImageBuffer(m_depthTexture[i]);
		}
		m_depthTexture.clear();
		for (int i = 0; i < m_vmaUniformBuffers.size(); i++)
		{
			BufferManager::destroyBuffer(m_vmaUniformBuffers[i]);
		}
		m_vmaUniformBuffers.clear();
	}
}