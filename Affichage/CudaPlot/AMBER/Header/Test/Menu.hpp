#ifndef __MENU__
#define __MENU__

#include "Scene.hpp"
#include "GameEngine.hpp"
#include "implot.h"

class Menu : public Scene, public ImguiBlock, public Behaviour
{
public:
	void load();
	void unload();
	void preRender(VulkanMisc* vM);
	void render(VulkanMisc* vM);
	void start();
	void fixedUpdate();
	void update();
	void stop();
	void onGUI();
private:
	ptrClass m_pc;
};

#endif //!__MENU__