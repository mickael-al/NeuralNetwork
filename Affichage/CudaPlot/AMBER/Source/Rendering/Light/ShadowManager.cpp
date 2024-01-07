#include "ShadowManager.hpp"
#include "GraphiquePipeline.hpp"
#include "RenderPass.hpp"
#include "ImageViewSwapChains.hpp"
#include "ColorResources.hpp"

namespace Ge
{	
	ShadowManager* ShadowManager::s_instance = nullptr;
	bool ShadowManager::initialize(VulkanMisc* vM, LightManager* lm)
	{
		vMisc = vM;
		m_lm = lm;
		Debug::INITSUCCESS("Shadow Manager");
		s_instance = this;
		if (!BufferManager::createBuffer(sizeof(ShadowMatrix), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_vmaUniformBuffers, vM->str_VulkanDeviceMisc))
		{
			Debug::Error("Echec de la creation d'un uniform buffer");
			return false;
		}
		ShadowMatrix pc_empty;
		pc_empty.splitDepth = 0.0f;
		pc_empty.projview = glm::mat4();
		memcpy(BufferManager::mapMemory(m_vmaUniformBuffers), &pc_empty, sizeof(ShadowMatrix));
		BufferManager::unMapMemory(m_vmaUniformBuffers);

		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = RenderPass::findDepthFormat(vM);
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 0;
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCreateInfo = {};
		renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		//renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		//renderPassCreateInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(vM->str_VulkanDeviceMisc->str_device, &renderPassCreateInfo, nullptr, &m_renderPass) != VK_SUCCESS)
		{
			Debug::Error("Echec de la creation du render pass");
			return false;
		}

		if (!createPipeline())
		{
			return false;
		}

		for (int i = 0; i < m_shadows.size(); i++)
		{
			m_shadows[i]->createFrameBuffer(m_renderPass);
		}
		
		updateDescriptor();
		return true;
	}

	ShadowManager* ShadowManager::getShadowManager()
	{
		return s_instance;
	}

	Shadow* ShadowManager::CreateShadow(LightData* light)
	{
		Shadow* s = new Shadow(m_renderPass,light, vMisc);
		m_shadows.push_back(s);		
		int countShadowImage = 0;
		for (int i = 0; i < m_shadows.size(); i++)
		{
			countShadowImage += m_shadows[i]->getImageView().size();
		}
		vMisc->str_VulkanDescriptor->shadowCount = countShadowImage;

		if (light->ubl->status == 1)
		{
			light->ubl->shadowID = shadowCubeMapCount;
			shadowCubeMapCount++;//SHADOW_MAP_CUBE_COUNT
		}
		else if (light->ubl->status == 0 || light->ubl->status == 2)
		{
			light->ubl->shadowID = shadowMapCount;
			shadowMapCount += light->ubl->status == 0 ? SHADOW_MAP_CASCADE_COUNT : SHADOW_MAP_SPOT_COUNT;
		}		
		else
		{
			Debug::Error("Spot light ubl not implemented light->ubl->status ShadowManager l122");
		}

		updateDescriptor();
		vMisc->str_VulkanDescriptor->recreateCommandBuffer = true;
		vMisc->str_VulkanDescriptor->recreateShadowPipeline = true;
		return s;
	}

	void ShadowManager::recreatePipeline()
	{
		vkDestroyPipelineCache(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_graphicsPipelineCache, nullptr);
		vkDestroyPipeline(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_pipelineLayout, nullptr);
		createPipeline();
	}

	void ShadowManager::initDescriptor(VulkanMisc* vM)
	{
		if (m_descriptor.size() == 0)
		{
			m_descriptor.push_back(new Descriptor(vM, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1));
			m_descriptor.push_back(new Descriptor(vM, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1));
		}
	}

	void ShadowManager::updateDirectionalShadow()
	{
		for (int i = 0; i < m_shadows.size(); i++)
		{
			if (m_shadows[i]->getLightData()->ubl->status == 0)
			{
				m_shadows[i]->mapMemory();
			}
		}
	}

	void ShadowManager::updateDescriptor()
	{
		std::vector<VkDescriptorImageInfo> imageInfo{};
		VkDescriptorImageInfo imageI{};
		if (m_shadows.size() == 0)
		{
			imageI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageI.imageView = vMisc->str_VulkanCommandeBufferMisc->str_nullTexture;
			imageI.sampler = vMisc->str_VulkanCommandeBufferMisc->str_nullTextureSampler;
			imageInfo.push_back(imageI);
			m_descriptor[0]->updateCount(vMisc, 1, imageInfo);
		}
		else
		{
			for (int i = 0; i < m_shadows.size(); i++)
			{
				for (int j = 0; j < m_shadows[i]->getImageView().size(); j++)
				{
					imageI.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
					imageI.imageView = m_shadows[i]->getImageView()[j];
					imageI.sampler = m_shadows[i]->getImageSampler();
					imageInfo.push_back(imageI);					
				}
			}
			m_descriptor[0]->updateCount(vMisc, imageInfo.size(), imageInfo);
		}		

		std::vector<VkDescriptorBufferInfo> updateInfo{};
		VkDescriptorBufferInfo uniformI;
		if (m_shadows.size() == 0)
		{
			uniformI.buffer = m_vmaUniformBuffers.buffer;
			uniformI.offset = 0;
			uniformI.range = sizeof(ShadowMatrix);
			updateInfo.push_back(uniformI);
			m_descriptor[1]->updateCount(vMisc, 1, updateInfo);
		}
		else
		{
			for (int i = 0; i < m_shadows.size(); i++)
			{
				std::vector<VkBuffer> buffers = m_shadows[i]->getUniformBuffers();
				for (int j = 0; j < buffers.size(); j++)
				{
					uniformI.buffer = buffers[j];
					uniformI.offset = 0;
					uniformI.range = sizeof(ShadowMatrix);
					updateInfo.push_back(uniformI);
				}
			}
			m_descriptor[1]->updateCount(vMisc, updateInfo.size(), updateInfo);
		}
	}

	bool ShadowManager::createPipeline()
	{
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
		ShaderElement VertShader = GraphiquePipeline::LoadShader("../Shader/Shadow_vs.spv", "main", vMisc->str_VulkanDeviceMisc->str_device, true, vMisc);
		ShaderElement FragShader = GraphiquePipeline::LoadShader("../Shader/Shadow_fs.spv", "main", vMisc->str_VulkanDeviceMisc->str_device, false, vMisc);
		shaderStages[0] = VertShader.shaderStageCreateInfo;
		shaderStages[1] = FragShader.shaderStageCreateInfo;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		std::array<VkVertexInputAttributeDescription, 2> attributeShadowDescriptions = Vertex::getShadowAttributeDescriptions();
		std::vector<VkVertexInputBindingDescription> bindingDescription = { Vertex::getBindingDescription(), PushConstants::getBindingDescription() };
		std::array<VkVertexInputAttributeDescription, 2> pushConstantsAttributeShadowDescriptions = PushConstants::getAttributeDescriptions(2);
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{ attributeShadowDescriptions[0], attributeShadowDescriptions[1], pushConstantsAttributeShadowDescriptions[0], pushConstantsAttributeShadowDescriptions[1] };

		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.flags = 0;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkExtent2D extentShadow{};
		extentShadow.height = TEXTURE_DIM;
		extentShadow.width = TEXTURE_DIM;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = extentShadow;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = TEXTURE_DIM;
		viewport.height = TEXTURE_DIM;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;
		viewportState.flags = 0;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.flags = 0;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;		

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = 0xf;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.attachmentCount = 0;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushConstantShadow);

		std::array<VkDescriptorSetLayout, 4> descriptorSetLayouts{ m_descriptor[1]->getDescriptorSetLayout(), vMisc->str_VulkanSwapChainMisc->str_descriptorSetLayout[1], vMisc->str_VulkanSwapChainMisc->str_descriptorSetLayout[2], vMisc->str_VulkanSwapChainMisc->str_descriptorSetLayout[3] };
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynamicStateEnables.data();
		dynamicState.dynamicStateCount = dynamicStateEnables.size();
		dynamicState.flags = 0;

		if (vkCreatePipelineLayout(vMisc->str_VulkanDeviceMisc->str_device, &pipelineLayoutInfo, nullptr, &m_graphiquePipelineElement.m_pipelineLayout) != VK_SUCCESS)
		{
			Debug::Error("Echec de la creation d'un pipeline layout");
			return false;
		}

		VkPipelineCacheCreateInfo cacheInfo{};
		cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		cacheInfo.pNext = NULL;
		cacheInfo.flags = 0;
		cacheInfo.initialDataSize = 0;
		cacheInfo.pInitialData = nullptr;

		if (vkCreatePipelineCache(vMisc->str_VulkanDeviceMisc->str_device, &cacheInfo, nullptr, &m_graphiquePipelineElement.m_graphicsPipelineCache) != VK_SUCCESS)
		{
			Debug::Error("Echec de la creation d'un cache pour le pipeline");
			return false;
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.layout = m_graphiquePipelineElement.m_pipelineLayout;
		pipelineInfo.renderPass = m_renderPass;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.pNext = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_graphicsPipelineCache, 1, &pipelineInfo, nullptr, &m_graphiquePipelineElement.m_graphicsPipeline) != VK_SUCCESS)
		{
			Debug::Error("Echec de la creation du pipeline graphique");
			return false;
		}

		GraphiquePipeline::DestroyShaderElement(vMisc->str_VulkanDeviceMisc->str_device, VertShader);
		GraphiquePipeline::DestroyShaderElement(vMisc->str_VulkanDeviceMisc->str_device, FragShader);
		return true;
	}

	void ShadowManager::RemoveShadow(Shadow* shadow)
	{
		m_shadows.erase(std::remove(m_shadows.begin(), m_shadows.end(), shadow), m_shadows.end());
		delete shadow;
		int countShadowImage = 0;
		for (int i = 0; i < m_shadows.size(); i++)
		{
			countShadowImage += m_shadows[i]->getImageView().size();
		}
		vMisc->str_VulkanDescriptor->shadowCount = countShadowImage;
		
		shadowCubeMapCount = 0;
		shadowMapCount = 0;
		for (int i = 0; i < m_shadows.size(); i++)
		{
			LightData* ld = m_shadows[i]->getLightData();
			if (ld->ubl->status == 1)
			{
				ld->ubl->shadowID = shadowCubeMapCount;
				shadowCubeMapCount++;
			}
			else
			{
				ld->ubl->shadowID = shadowMapCount;
				shadowMapCount++;
			}
		}
		m_lm->mapMemory();
		updateDescriptor();
		vMisc->str_VulkanDescriptor->recreateCommandBuffer = true;		
		vMisc->str_VulkanDescriptor->recreateShadowPipeline = true;
	}

	const std::vector<Shadow*> & ShadowManager::getShadows() const
	{
		return m_shadows;
	}

	GraphiquePipelineElement ShadowManager::getGraphiquePipelineElement() const
	{
		return m_graphiquePipelineElement;
	}

	VkRenderPass ShadowManager::getRenderPass() const
	{
		return m_renderPass;
	}

	void ShadowManager::release()
	{
		for (int i = 0; i < m_shadows.size(); i++)
		{
			delete m_shadows[i];
		}
		m_shadows.clear();
		BufferManager::destroyBuffer(m_vmaUniformBuffers);
		vkDestroyPipelineCache(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_graphicsPipelineCache, nullptr);
		vkDestroyPipeline(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(vMisc->str_VulkanDeviceMisc->str_device, m_graphiquePipelineElement.m_pipelineLayout, nullptr);		
		vkDestroyRenderPass(vMisc->str_VulkanDeviceMisc->str_device, m_renderPass, nullptr);
		for (int i = 0; i < m_descriptor.size(); i++)
		{
			delete m_descriptor[i];
		}
		m_descriptor.clear();
		Debug::RELEASESUCCESS("Shadow Manager");
	}
}