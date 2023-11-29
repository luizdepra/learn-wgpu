use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    window: Window,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // Instance is a handle to the GPU. Used to create Adapters and Surfaces.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Vulkan + Metal + DX12 + WebGPU
            ..Default::default()
        });

        // Part fo the window that we draw to.
        // Surface should live as long the window. Should be safe because everything is inside State.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        // A handle to the actual graphic card.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::default(), // LowPower (integrated GPU) nad HighPerformance (video cadrd)
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WeGL doesn't support all features.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb()) // We'll use shader code that assumes an sRGB surface texture.
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // Texture will be used to write to the screen.
            format: surface_format, // The format of the texture that will be stored
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // How to sync the surface with the display.
            alpha_mode: surface_caps.alpha_modes[0], // Don't know. Probably about transparent windows.
            view_formats: vec![], // List of texture formats that can be used to create TextureViews.
        };
        surface.configure(&device, &config);

        // Color to clear the screen.
        let clear_color = wgpu::Color::BLACK;

        // Creates a shader module with out file.
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // Defines how to interepret out vertices.
            primitive: wgpu::PrimitiveState {
                // PrimitiveTopology::TriangleList means that every 3 vertices will be one triangle.
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // Means that a triangle is facing foward if its vertices are arranged in a conter-clockwise direction.
                cull_mode: Some(wgpu::Face::Back), // Tells to cull/not render if the triangles are facing back.
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,                         // Use all samples.
                alpha_to_coverage_enabled: false, // Related with anti-aliasing.
            },
            multiview: None,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let r = position.x as f64 / self.size.width as f64;
                let g = position.y as f64 / self.size.height as f64;
                let b = r * g;

                self.clear_color = wgpu::Color { r, g, b, a: 1.0 };
                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // Controls how to interact with the surface texture.
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Builds a command buffer that we can send to the gpu.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                // Where we gonna draw colors to.
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets to.
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,          // Texture to save the color to.
                        resolve_target: None, // Texture that will receive the color. Should be None unless when using multisampling.
                        ops: wgpu::Operations {
                            // Tells what to do with the color.
                            // Load tell how to handle colors stored from previous frame. We will clear the screen with a bluish color.
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            // Tell that we want to store the color in our screen texture.
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw(0..3, 0..1); // Draw 3 vertices and 1 instance.
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if it is lost.
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // If the system is out of memory, quit.
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // Other errors (Outdated, Timeout) should be resolved by the next frame.
                Err(e) => eprintln!("{:?}", e),
            }
        }

        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}
