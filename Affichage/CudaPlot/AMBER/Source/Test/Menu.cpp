#include "Menu.hpp"

void Menu::load()
{
	m_pc = GameEngine::getPtrClass();
	m_pc.hud->addBlockUI(this);
	
}

void Menu::unload()
{    
	m_pc.hud->removeBlockUI(this);
}

void Menu::start()
{

}

void Menu::fixedUpdate()
{

}

void Menu::update()
{

}

void Menu::stop()
{

}

void Menu::onGUI()
{

}

void Menu::preRender(VulkanMisc* vM)
{

}

void Menu::render(VulkanMisc* vM)
{
    if (ImPlot::BeginPlot("Xor Test", "X", "Y")) {
        // Create some sample data
        const double x[] = { 1.0, -1.0};
        const double y[] = { 1.0, -1.0};
        const double x1[] = {1.0, -1.0};
        const double y1[] = {-1.0, 1.0};

        // Plot the scatter plot
        ImPlot::PlotScatter("False", x, y, 2);
        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 6, ImPlot::GetColormapColor(1), IMPLOT_AUTO, ImPlot::GetColormapColor(1));
        ImPlot::PlotScatter("True", x1, y1, 2);
        ImPlot::PlotLine("Line", &x1[0], &y1[0], 2, 1, 0, sizeof(double));

        // End the plot
        ImPlot::EndPlot();
    }

}