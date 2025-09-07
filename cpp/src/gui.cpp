#include <iostream>
#include <fstream>
#include <vector>
#include "imgui_internal.h"
#include "nine_or_null/fft.h"

// example code begin
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>
// example code end

#include "nine_or_null/nine_or_null.h"
#include "spectrogram.h"

auto fmt = fmt_precision();
const double step = double(1.0 / _DIV_PRECISION);
const double minor_min = -1000.0;
const double minor_max = 1000.0;
const double major_min = -1000000.0;
const double major_max = 1000000.0;
#define CTRL_DOUBLE(label, var, minor_or_major) \
    ImGui::DragScalar( \
        label, \
        ImGuiDataType_Double, \
        &var, \
        step, \
        &minor_or_major##_min, \
        &minor_or_major##_max, \
        fmt.c_str(), \
        ImGuiSliderFlags_AlwaysClamp \
    )
#define LOCK_BEAT(label, var, minor_or_major) \
    CTRL_DOUBLE(label, var, minor_or_major); \
    if (!ImGui::IsItemActive() && ImGui::IsItemDeactivatedAfterEdit()) { \
        var = BeatFraction(var); \
    }
#define LOCK_VALUE(label, var, minor_or_major) \
    CTRL_DOUBLE(label, var, minor_or_major); \
    if (!ImGui::IsItemActive() && ImGui::IsItemDeactivatedAfterEdit()) { \
        var = to_precision(var); \
    }
#define EVENT_VECTOR_ADD_BUTTON(label_stem, text, vec, index) \
    ImGui::PushID((std::string(label_stem) + "##AddAbove##" + std::to_string(index)).c_str()); \
    if (ImGui::Button(text)) { \
        if (index == 0) { \
            vec.emplace(vec.begin(), 0, 0); \
        } else { \
            vec.emplace(vec.begin() + index, vec[index-1].beat, vec[index-1].value);\
        } \
    } \
    ImGui::PopID()
#define EVENT_VECTOR_DELETE_BUTTON(label_stem, text, vec, index) \
    ImGui::PushID((std::string(label_stem) + "##Delete##" + std::to_string(index)).c_str()); \
    if (ImGui::Button(text)) { \
        vec.erase(vec.begin() + index); \
    } \
    ImGui::PopID()

// example code begin
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int imgui_main()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow((int)(1280), (int)(800), "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io; 
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(1);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    bool show_9on_window = true;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    GLuint texture;
    nine_or_null::Wave wave;
    std::ifstream fp;
    fp.open("C:\\Users\\telpi\\Documents\\GitHub\\nine-or-null\\cpp\\src\\ADONIS.wav", std::ios::in | std::ios::binary);
    fp >> wave;
    fp.close();
    std::cout << wave;

    nine_or_null::WaveData data;
    wave.fill(data, 0);
    
    float sample_rate = wave.wave_fmt_chunk().nSamplesPerSec;
    float t_start = 15.0f;
    float t_end = 20.0f;
    size_t window_size = 8;
    size_t stride = 441;
    size_t reduce_rate = 12;

    nine_or_null::WaveData initial_data(
        data.begin() + size_t(sample_rate * t_start),
        data.begin() + size_t(sample_rate * t_end)
    );
    Spectrogram gram = create_spectrogram(
        initial_data, 
        window_size,
        stride,
        reduce_rate
    );
    prepare_texture(texture);
    bool gram_success = update_texture(
        texture,
        gram
    );


    // 9oN state
    Simfile simfile;
    fp.open("C:\\Users\\telpi\\Documents\\GitHub\\nine-or-null\\cpp\\src\\ADONIS.ssc", std::ios::in);
    fp >> simfile;
    fp.close();
    std::cout << simfile;

    nine_or_null::StackedLocalResponse slr;
    slr.metadata(wave);
    slr.stack_local_response(data, simfile);
    slr.update_texture(texture);
    

    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");

            bool needs_update = false;

            static float t_start_new = t_start;
            static float t_end_new = t_end;
            static int window_size_new = window_size;
            static float stride_new_f = stride / sample_rate;
            static int stride_new = 1;
            static int reduce_rate_new = reduce_rate;
            
            ImGui::SliderFloat("t_0", &t_start_new, 0.0f, wave.length(), "%0.3f", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                if (t_start_new != t_start && t_start_new < t_end) {
                    t_start = t_start_new;
                    needs_update = true;
                }
                else {
                    t_start_new = t_start;
                }
            }

            ImGui::SliderFloat("t_f", &t_end_new, 0.0f, wave.length(), "%0.3f", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                if (t_end_new != t_end && t_end_new > t_start) {
                    t_end = t_end_new;
                    needs_update = true;
                }
                else {
                    t_end_new = t_end;
                }
            }

            ImGui::SliderInt("Window size", &window_size_new, 4, 16, "%d bits", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                if (window_size_new != window_size) {
                    window_size = window_size_new;
                    needs_update = true;
                }
            }

            ImGui::SliderFloat("Stride", &stride_new_f, 0.001f, 1.0f, "%0.6f seconds", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                stride_new_f = (stride_new_f < 0.001f) ? 0.001f : ((stride_new_f > 1.0f) ? 1.0f : stride_new_f);
                stride_new = int(stride_new_f * sample_rate);
                if (stride_new != stride) {
                    stride = stride_new;
                    stride_new_f = stride_new / sample_rate;
                    needs_update = true;
                }
                else {
                    stride_new_f = stride_new / sample_rate;
                }
            }

            ImGui::InputInt("Reduce sampling rate", &reduce_rate_new, 1, 10);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                if (
                    reduce_rate_new != reduce_rate &&
                    reduce_rate_new >= 1 &&
                    reduce_rate_new <= 100
                ) {
                    reduce_rate = reduce_rate_new;
                    needs_update = true;
                }
                else {
                    reduce_rate_new = reduce_rate;
                }
            }

            if (needs_update) {
                nine_or_null::WaveData partial(
                    data.begin() + size_t(sample_rate * t_start),
                    data.begin() + size_t(sample_rate * t_end)
                );
                gram = create_spectrogram(
                    partial, 
                    window_size,
                    stride,
                    reduce_rate
                );
                gram_success = update_texture(
                    texture,
                    gram
                );
            }
            if (gram_success) {
                ImGui::Text("pointer = %x", texture);
                ImGui::Text("size = %zu x %zu", gram.width, gram.height);
                auto pos = ImGui::GetCursorScreenPos();
                ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(gram.width, gram.height));
                if (ImGui::IsItemHovered())
                {
                    size_t x = size_t(io.MousePos.x - pos.x);
                    size_t y = size_t(io.MousePos.y - pos.y);
                    float x_center_time = float(x * stride) / float(sample_rate);
                    float y_freq_tap = nine_or_null::tap(
                        y,
                        (2 << window_size) + 1,
                        sample_rate,
                        reduce_rate
                    );
                    ImGui::BeginTooltip();
                    ImGui::Text("x = %zu (t = %0.6f)", x, x_center_time);
                    ImGui::Text("y = %zu (f = %0.3f)", y, y_freq_tap);
                    ImGui::Text("z = %0.3f", gram.data[y * gram.width + x]);
                    ImGui::EndTooltip();
                }
            }
            else {
                ImGui::Text("Couldn't make spectrogram");
            }
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }
        
        if (show_9on_window)
        {
            ImGuiWindowFlags window_flags = (
                ImGuiWindowFlags_HorizontalScrollbar                
            );
            {
                ImGui::BeginGroup();
                float w = 80; // (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.3f;

                ImGui::SetNextItemWidth(w);
                LOCK_VALUE("Beat 0 Offset", simfile.offset, minor);

                ImGui::Text("BPMs");
                float side_panel_height = (ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y - ImGui::GetTextLineHeightWithSpacing()) * 0.5f;
                {
                    ImGui::BeginChild("BPMList", ImVec2(300, side_panel_height), ImGuiChildFlags_Borders, window_flags);
                    for (int i = 0; i < simfile.bpms.size(); i++)
                    {
                        EVENT_VECTOR_ADD_BUTTON("##BPM", "++ ^^", simfile.bpms, i);
                        ImVec2 size = ImGui::GetItemRectSize();
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(w);
                        LOCK_BEAT(("##BPM##Beat##" + std::to_string(i)).c_str(), simfile.bpms[i].beat, major);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(w);
                        LOCK_VALUE(("##BPM##BPM##" + std::to_string(i)).c_str(), simfile.bpms[i].value, major);
                        ImGui::SameLine();
                        EVENT_VECTOR_DELETE_BUTTON("##BPM", "<< --", simfile.bpms, i);
                    }
                    EVENT_VECTOR_ADD_BUTTON("##BPM", "++ ..", simfile.bpms, simfile.bpms.size());
                    ImGui::EndChild();
                }
                ImGui::Text("Stops");
                {
                    ImGui::BeginChild("StopList", ImVec2(300, side_panel_height), ImGuiChildFlags_Borders, window_flags);
                    for (int i = 0; i < simfile.stops.size(); i++)
                    {
                        EVENT_VECTOR_ADD_BUTTON("##Stop", "++ ^^", simfile.stops, i);
                        ImVec2 size = ImGui::GetItemRectSize();
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(w);
                        LOCK_BEAT(("##Stop##Beat##" + std::to_string(i)).c_str(), simfile.stops[i].beat, major);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(w);
                        LOCK_VALUE(("##Stop##Stop##" + std::to_string(i)).c_str(), simfile.stops[i].value, major);
                        ImGui::SameLine();
                        EVENT_VECTOR_DELETE_BUTTON("##Stop", "<< --", simfile.stops, i);
                    }
                    EVENT_VECTOR_ADD_BUTTON("##Stop", "++ ^^", simfile.stops, simfile.stops.size());
                    ImGui::EndChild();
                }
                ImGui::EndGroup();
            }
            ImGui::SameLine();
            { 
                ImGui::BeginGroup();
                ImGui::Text("Stacked Local Response");
                if (gram_success) {
                    ImGui::Text("pointer = %x", texture);
                    ImGui::Text("size = %zu x %zu", slr.slr[0].size(), slr.slr.size());
                    auto pos = ImGui::GetCursorScreenPos();
                    ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(slr.slr[0].size(), slr.slr.size()));
                    if (ImGui::IsItemHovered())
                    {
                        size_t x = size_t(io.MousePos.x - pos.x);
                        size_t y = size_t(io.MousePos.y - pos.y);
                        float x_center_time = float(x * stride) / float(sample_rate);
                        float y_freq_tap = nine_or_null::tap(
                            y,
                            (2 << window_size) + 1,
                            sample_rate,
                            reduce_rate
                        );
                        ImGui::BeginTooltip();
                        ImGui::Text("x = %zu (t = %0.6f)", x, x_center_time);
                        ImGui::Text("y = %zu (f = %0.3f)", y, y_freq_tap);
                        ImGui::Text("z = %0.3f", gram.data[y * gram.width + x]);
                        ImGui::EndTooltip();
                    }
                }
                ImGui::EndGroup();
            }

        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    printf("DestroyContext()\n");

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
// example code end

int main(int argc, const char* argv[]) {
    std::cout << "Hello World? (GUI) " << int(nine_or_null::do_the_thing() * 86400) << std::endl;
    
    return imgui_main();
}