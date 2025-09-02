"use client";

import { Camera, Mesh, Plane, Program, Renderer, Texture, Transform } from "ogl";
import { useEffect, useRef, useState } from "react";

type GL = Renderer["gl"];

function debounce<T extends (...args: any[]) => void>(func: T, wait: number) {
  let timeout: number;
  return function (this: any, ...args: Parameters<T>) {
    window.clearTimeout(timeout);
    timeout = window.setTimeout(() => func.apply(this, args), wait);
  };
}

function lerp(p1: number, p2: number, t: number): number {
  return p1 + (p2 - p1) * t;
}

function autoBind(instance: any): void {
  const proto = Object.getPrototypeOf(instance);
  Object.getOwnPropertyNames(proto).forEach((key) => {
    if (key !== "constructor" && typeof instance[key] === "function") {
      instance[key] = instance[key].bind(instance);
    }
  });
}

function getFontSize(font: string): number {
  const match = font.match(/(\d+)px/);
  return match ? parseInt(match[1], 10) : 30;
}

function createCardTexture(
  gl: GL,
  icon: string,
  text: string,
  font: string = "bold 18px sans-serif",
  color: string = "#ffffff",
  backgroundColor: string = "#222222",
  description: string = ""
): { texture: Texture; width: number; height: number } {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not get 2d context");

  // Set canvas dimensions - increased height to fit full description
  const width = 300;
  const height = 220;
  canvas.width = width;
  canvas.height = height;

  // Draw white background with rounded corners
  context.fillStyle = "#FFFFFF";
  const borderRadius = 12;
  context.beginPath();
  context.moveTo(borderRadius, 0);
  context.lineTo(width - borderRadius, 0);
  context.quadraticCurveTo(width, 0, width, borderRadius);
  context.lineTo(width, height - borderRadius);
  context.quadraticCurveTo(width, height, width - borderRadius, height);
  context.lineTo(borderRadius, height);
  context.quadraticCurveTo(0, height, 0, height - borderRadius);
  context.lineTo(0, borderRadius);
  context.quadraticCurveTo(0, 0, borderRadius, 0);
  context.closePath();
  context.fill();

  // Draw icon at the top center
  const iconSize = 36;
  const iconX = width / 2;
  const iconY = 40;
  
  // Create a rich purple gradient for the icon (SVG-like appearance)
  const gradient = context.createLinearGradient(iconX - iconSize/2, iconY - iconSize/2, iconX + iconSize/2, iconY + iconSize/2);
  // Enhanced purple gradient that renders better with lightning bolt symbols
  gradient.addColorStop(0, "#C4B5FD"); // Purple-300
  gradient.addColorStop(0.5, "#A78BFA"); // Purple-400
  gradient.addColorStop(1, "#8B5CF6"); // Purple-500
  
  // Draw icon with enhanced gradient fill, no background or border
  context.font = "bold 28px sans-serif"; // Slightly larger for better gradient visibility
  context.fillStyle = gradient;
  context.strokeStyle = gradient; // Use gradient for stroke too for better definition
  context.lineWidth = 0.5;
  context.textBaseline = "middle";
  context.textAlign = "center";
  context.fillText(icon, iconX, iconY);
  // For special symbols, add subtle stroke for better definition
  if (icon === "↯" || icon === "⥈" || icon === "⤑" || icon === "⟩⟩") {
    context.strokeText(icon, iconX, iconY);
  }

  // Draw title text centered under icon
  context.font = "bold 16px sans-serif"; 
  context.fillStyle = "#333333"; // Dark text color
  context.textAlign = "center";
  
  // Handle text that's too long by using multiple lines
  const maxTitleWidth = width - 40; // Leave some margin
  let titleY = height - 120; // Position title higher to leave room for full description
  
  if (context.measureText(text).width > maxTitleWidth) {
    // Break title into two lines if needed
    const words = text.split(' ');
    let line1 = '';
    let line2 = '';
    
    // Try to distribute words evenly between two lines
    const middleIndex = Math.ceil(words.length / 2);
    line1 = words.slice(0, middleIndex).join(' ');
    line2 = words.slice(middleIndex).join(' ');
    
    context.fillText(line1, width / 2, titleY);
    context.fillText(line2, width / 2, titleY + 22);
    titleY += 44; // Update the starting position for description
  } else {
    context.fillText(text, width / 2, titleY);
    titleY += 22; // Update the starting position for description
  }

  // Draw description text with word wrapping to show full text
  if (description) {
    context.font = "14px sans-serif";
    context.fillStyle = "#666677"; // Gray text color
    context.textAlign = "center";
    
    const maxDescWidth = width - 50; // Leave margin for description
    const lineHeight = 18; // Line height for description text
    const words = description.split(' ');
    let line = '';
    let descY = titleY + 10; // Start description after title with some spacing
    
    // Word wrap algorithm for description
    for (let i = 0; i < words.length; i++) {
      const testLine = line + words[i] + ' ';
      const metrics = context.measureText(testLine);
      
      if (metrics.width > maxDescWidth && i > 0) {
        context.fillText(line, width / 2, descY);
        line = words[i] + ' ';
        descY += lineHeight;
      } else {
        line = testLine;
      }
    }
    
    // Draw the last line
    context.fillText(line, width / 2, descY);
  }

  // Create texture
  const texture = new Texture(gl, { generateMipmaps: false });
  texture.image = canvas;
  return { texture, width, height };
}

interface ScreenSize {
  width: number;
  height: number;
}

interface Viewport {
  width: number;
  height: number;
}

interface MediaProps {
  geometry: Plane;
  gl: GL;
  icon: string;
  text: string;
  description?: string;
  index: number;
  length: number;
  renderer: Renderer;
  scene: Transform;
  screen: ScreenSize;
  viewport: Viewport;
  bend: number;
  textColor: string;
  backgroundColor?: string;
  borderRadius?: number;
  font?: string;
}

class Media {
  extra: number = 0;
  geometry: Plane;
  gl: GL;
  icon: string;
  text: string;
  description: string;
  index: number;
  length: number;
  renderer: Renderer;
  scene: Transform;
  screen: ScreenSize;
  viewport: Viewport;
  bend: number;
  textColor: string;
  backgroundColor: string;
  borderRadius: number;
  font: string;
  program!: Program;
  plane!: Mesh;
  scale!: number;
  padding!: number;
  width!: number;
  widthTotal!: number;
  x!: number;
  speed: number = 0;
  isBefore: boolean = false;
  isAfter: boolean = false;

  constructor({
    geometry,
    gl,
    icon,
    text,
    description = "",
    index,
    length,
    renderer,
    scene,
    screen,
    viewport,
    bend,
    textColor,
    backgroundColor = "#222222",
    borderRadius = 0,
    font = "bold 24px sans-serif",
  }: MediaProps) {
    this.geometry = geometry;
    this.gl = gl;
    this.icon = icon;
    this.text = text;
    this.description = description;
    this.index = index;
    this.length = length;
    this.renderer = renderer;
    this.scene = scene;
    this.screen = screen;
    this.viewport = viewport;
    this.bend = bend;
    this.textColor = textColor;
    this.backgroundColor = backgroundColor;
    this.borderRadius = borderRadius;
    this.font = font;
    this.createShader();
    this.createMesh();
    this.onResize();
  }

  createShader() {
    // Use a consistent color for all cards regardless of their individual backgroundColor
    const purpleColor = "#8B5CF6"; // Always use purple for consistency
    const { texture, width, height } = createCardTexture(
      this.gl, 
      this.icon, 
      this.text, 
      this.font, 
      this.textColor,
      purpleColor, // Always use the same purple color for all icons
      this.description
    );
    
    this.program = new Program(this.gl, {
      depthTest: false,
      depthWrite: false,
      vertex: `
        precision highp float;
        attribute vec3 position;
        attribute vec2 uv;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform float uTime;
        uniform float uSpeed;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          vec3 p = position;
          p.z = (sin(p.x * 4.0 + uTime) * 1.5 + cos(p.y * 2.0 + uTime) * 1.5) * (0.1 + uSpeed * 0.5);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragment: `
        precision highp float;
        uniform vec2 uImageSizes;
        uniform vec2 uPlaneSizes;
        uniform sampler2D tMap;
        uniform float uBorderRadius;
        varying vec2 vUv;
        
        float roundedBoxSDF(vec2 p, vec2 b, float r) {
          vec2 d = abs(p) - b;
          return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - r;
        }
        
        void main() {
          vec2 ratio = vec2(
            min((uPlaneSizes.x / uPlaneSizes.y) / (uImageSizes.x / uImageSizes.y), 1.0),
            min((uPlaneSizes.y / uPlaneSizes.x) / (uImageSizes.y / uImageSizes.x), 1.0)
          );
          vec2 uv = vec2(
            vUv.x * ratio.x + (1.0 - ratio.x) * 0.5,
            vUv.y * ratio.y + (1.0 - ratio.y) * 0.5
          );
          vec4 color = texture2D(tMap, uv);
          
          float d = roundedBoxSDF(vUv - 0.5, vec2(0.5 - uBorderRadius), uBorderRadius);
          
          // Smooth antialiasing for edges
          float edgeSmooth = 0.002;
          float alpha = 1.0 - smoothstep(-edgeSmooth, edgeSmooth, d);
          
          gl_FragColor = vec4(color.rgb, alpha);
        }
      `,
      uniforms: {
        tMap: { value: texture },
        uPlaneSizes: { value: [0, 0] },
        uImageSizes: { value: [width, height] },
        uSpeed: { value: 0 },
        uTime: { value: 100 * Math.random() },
        uBorderRadius: { value: this.borderRadius },
      },
      transparent: true,
    });
  }

  createMesh() {
    this.plane = new Mesh(this.gl, {
      geometry: this.geometry,
      program: this.program,
    });
    this.plane.setParent(this.scene);
  }

  update(scroll: { current: number; last: number }, direction: "right" | "left") {
    this.plane.position.x = this.x - scroll.current - this.extra;

    const x = this.plane.position.x;
    const H = this.viewport.width / 2;

    if (this.bend === 0) {
      this.plane.position.y = 0;
      this.plane.rotation.z = 0;
    } else {
      const B_abs = Math.abs(this.bend);
      const R = (H * H + B_abs * B_abs) / (2 * B_abs);
      const effectiveX = Math.min(Math.abs(x), H);

      const arc = R - Math.sqrt(R * R - effectiveX * effectiveX);
      if (this.bend > 0) {
        this.plane.position.y = -arc;
        this.plane.rotation.z = -Math.sign(x) * Math.asin(effectiveX / R);
      } else {
        this.plane.position.y = arc;
        this.plane.rotation.z = Math.sign(x) * Math.asin(effectiveX / R);
      }
    }

    this.speed = scroll.current - scroll.last;
    this.program.uniforms.uTime.value += 0.04;
    this.program.uniforms.uSpeed.value = this.speed;

    const planeOffset = this.plane.scale.x / 2;
    const viewportOffset = this.viewport.width / 2;
    this.isBefore = this.plane.position.x + planeOffset < -viewportOffset;
    this.isAfter = this.plane.position.x - planeOffset > viewportOffset;
    if (direction === "right" && this.isBefore) {
      this.extra -= this.widthTotal;
      this.isBefore = this.isAfter = false;
    }
    if (direction === "left" && this.isAfter) {
      this.extra += this.widthTotal;
      this.isBefore = this.isAfter = false;
    }
  }

  onResize({ screen, viewport }: { screen?: ScreenSize; viewport?: Viewport } = {}) {
    if (screen) this.screen = screen;
    if (viewport) {
      this.viewport = viewport;
      if (this.plane.program.uniforms.uViewportSizes) {
        this.plane.program.uniforms.uViewportSizes.value = [this.viewport.width, this.viewport.height];
      }
    }
    this.scale = this.screen.height / 1200;
    
    // Calculate aspect ratio to match our new texture (300x220)
    const aspectRatio = 300 / 220; // Width / Height of our texture
    
    // Set height first - adjusted for new card size
    this.plane.scale.y = (this.viewport.height * (750 * this.scale)) / this.screen.height;
    // Calculate width based on the aspect ratio
    this.plane.scale.x = this.plane.scale.y * aspectRatio;
    
    this.plane.program.uniforms.uPlaneSizes.value = [this.plane.scale.x, this.plane.scale.y];
    this.padding = 3.5; // Adjust padding between cards
    this.width = this.plane.scale.x + this.padding;
    this.widthTotal = this.width * this.length;
    this.x = this.width * this.index;
  }
}

interface GalleryItem {
  icon: string;
  text: string;
  description?: string;
  backgroundColor?: string;
}

interface AppConfig {
  items?: GalleryItem[];
  bend?: number;
  textColor?: string;
  backgroundColor?: string;
  borderRadius?: number;
  font?: string;
  scrollSpeed?: number;
  scrollEase?: number;
}

class App {
  container: HTMLElement;
  scrollSpeed: number;
  scroll: {
    ease: number;
    current: number;
    target: number;
    last: number;
    position?: number;
  };
  onCheckDebounce: (...args: any[]) => void;
  renderer!: Renderer;
  gl!: GL;
  camera!: Camera;
  scene!: Transform;
  planeGeometry!: Plane;
  medias: Media[] = [];
  mediasImages: GalleryItem[] = [];
  screen!: { width: number; height: number };
  viewport!: { width: number; height: number };
  raf: number = 0;

  boundOnResize!: () => void;
  boundOnWheel!: (e: Event) => void;
  boundOnTouchDown!: (e: MouseEvent | TouchEvent) => void;
  boundOnTouchMove!: (e: MouseEvent | TouchEvent) => void;
  boundOnTouchUp!: () => void;

  isDown: boolean = false;
  start: number = 0;

  constructor(
    container: HTMLElement,
    {
      items,
      bend = 1,
      textColor = "#ffffff",
      backgroundColor = "#222222",
      borderRadius = 0,
      font = "bold 24px sans-serif",
      scrollSpeed = 2,
      scrollEase = 0.05,
    }: AppConfig
  ) {
    document.documentElement.classList.remove("no-js");
    this.container = container;
    this.scrollSpeed = scrollSpeed;
    this.scroll = { ease: scrollEase, current: 0, target: 0, last: 0 };
    this.onCheckDebounce = debounce(this.onCheck.bind(this), 200);
    this.createRenderer();
    this.createCamera();
    this.createScene();
    this.onResize();
    this.createGeometry();
    this.createMedias(items, bend, textColor, backgroundColor, borderRadius, font);
    this.update();
    this.addEventListeners();
  }

  createRenderer() {
    this.renderer = new Renderer({ 
      alpha: true,
      antialias: true,
      dpr: Math.min(window.devicePixelRatio || 1, 2)
    });
    this.gl = this.renderer.gl;
    this.gl.clearColor(0, 0, 0, 0);
    this.container.appendChild(this.renderer.gl.canvas as HTMLCanvasElement);
  }

  createCamera() {
    this.camera = new Camera(this.gl);
    this.camera.fov = 45;
    this.camera.position.z = 20;
  }

  createScene() {
    this.scene = new Transform();
  }

  createGeometry() {
    this.planeGeometry = new Plane(this.gl, {
      heightSegments: 50,
      widthSegments: 100,
    });
  }

  createMedias(
    items: GalleryItem[] | undefined,
    bend: number = 1,
    textColor: string,
    backgroundColor: string,
    borderRadius: number,
    font: string
  ) {
    const defaultItems = [
      { icon: "✓", text: "Verifiable Computation" },
      { icon: "⚡", text: "Model Sharding" },
      { icon: "🌐", text: "Decentralized Network" },
      { icon: "⚡", text: "Lightning Fast" },
      { icon: "🔒", text: "Privacy Preserving" },
      { icon: "💰", text: "Cost Efficient" },
    ];
    
    const galleryItems = items && items.length ? items : defaultItems;
    this.mediasImages = galleryItems; // No longer doubling items
    
    this.medias = this.mediasImages.map((data, index) => {
      return new Media({
        geometry: this.planeGeometry,
        gl: this.gl,
        icon: data.icon,
        text: data.text,
        description: data.description,
        index,
        length: this.mediasImages.length,
        renderer: this.renderer,
        scene: this.scene,
        screen: this.screen,
        viewport: this.viewport,
        bend,
        textColor,
        backgroundColor: data.backgroundColor || backgroundColor,
        borderRadius,
        font,
      });
    });
  }

  onTouchDown(e: MouseEvent | TouchEvent) {
    this.isDown = true;
    this.scroll.position = this.scroll.current;
    this.start = "touches" in e ? e.touches[0].clientX : e.clientX;
  }

  onTouchMove(e: MouseEvent | TouchEvent) {
    if (!this.isDown) return;
    const x = "touches" in e ? e.touches[0].clientX : e.clientX;
    const distance = (this.start - x) * (this.scrollSpeed * 0.025);
    this.scroll.target = (this.scroll.position ?? 0) + distance;
  }

  onTouchUp() {
    this.isDown = false;
    this.onCheck();
  }

  onWheel(e: Event) {
    const wheelEvent = e as WheelEvent;
    const delta = wheelEvent.deltaY || (wheelEvent as any).wheelDelta || (wheelEvent as any).detail;
    this.scroll.target += (delta > 0 ? this.scrollSpeed : -this.scrollSpeed) * 0.2;
    this.onCheckDebounce();
  }

  onCheck() {
    if (!this.medias || !this.medias[0]) return;
    const width = this.medias[0].width;
    const itemIndex = Math.round(Math.abs(this.scroll.target) / width);
    const item = width * itemIndex;
    this.scroll.target = this.scroll.target < 0 ? -item : item;
  }

  onResize() {
    this.screen = {
      width: this.container.clientWidth,
      height: this.container.clientHeight,
    };
    this.renderer.setSize(this.screen.width, this.screen.height);
    this.camera.perspective({
      aspect: this.screen.width / this.screen.height,
    });
    const fov = (this.camera.fov * Math.PI) / 180;
    const height = 2 * Math.tan(fov / 2) * this.camera.position.z;
    const width = height * this.camera.aspect;
    this.viewport = { width, height };
    if (this.medias) {
      this.medias.forEach((media) => media.onResize({ screen: this.screen, viewport: this.viewport }));
    }
  }

  update() {
    this.scroll.current = lerp(this.scroll.current, this.scroll.target, this.scroll.ease);
    const direction = this.scroll.current > this.scroll.last ? "right" : "left";
    if (this.medias) {
      this.medias.forEach((media) => media.update(this.scroll, direction));
    }
    this.renderer.render({ scene: this.scene, camera: this.camera });
    this.scroll.last = this.scroll.current;
    this.raf = window.requestAnimationFrame(this.update.bind(this));
  }

  addEventListeners() {
    this.boundOnResize = this.onResize.bind(this);
    this.boundOnWheel = this.onWheel.bind(this);
    this.boundOnTouchDown = this.onTouchDown.bind(this);
    this.boundOnTouchMove = this.onTouchMove.bind(this);
    this.boundOnTouchUp = this.onTouchUp.bind(this);
    window.addEventListener("resize", this.boundOnResize);
    window.addEventListener("mousewheel", this.boundOnWheel);
    window.addEventListener("wheel", this.boundOnWheel);
    window.addEventListener("mousedown", this.boundOnTouchDown);
    window.addEventListener("mousemove", this.boundOnTouchMove);
    window.addEventListener("mouseup", this.boundOnTouchUp);
    window.addEventListener("touchstart", this.boundOnTouchDown);
    window.addEventListener("touchmove", this.boundOnTouchMove);
    window.addEventListener("touchend", this.boundOnTouchUp);
  }

  destroy() {
    window.cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.boundOnResize);
    window.removeEventListener("mousewheel", this.boundOnWheel);
    window.removeEventListener("wheel", this.boundOnWheel);
    window.removeEventListener("mousedown", this.boundOnTouchDown);
    window.removeEventListener("mousemove", this.boundOnTouchMove);
    window.removeEventListener("mouseup", this.boundOnTouchUp);
    window.removeEventListener("touchstart", this.boundOnTouchDown);
    window.removeEventListener("touchmove", this.boundOnTouchMove);
    window.removeEventListener("touchend", this.boundOnTouchUp);
    if (this.renderer && this.renderer.gl && this.renderer.gl.canvas.parentNode) {
      this.renderer.gl.canvas.parentNode.removeChild(this.renderer.gl.canvas as HTMLCanvasElement);
    }
  }
}

interface CircularGalleryProps {
  items?: GalleryItem[];
  bend?: number;
  textColor?: string;
  backgroundColor?: string;
  borderRadius?: number;
  font?: string;
  scrollSpeed?: number;
  scrollEase?: number;
}

const CircularGallery = ({
  items,
  bend = 3,
  textColor = "#ffffff",
  backgroundColor = "#222222",
  borderRadius = 0.05,
  font = "bold 28px sans-serif", // Increased font size
  scrollSpeed = 2,
  scrollEase = 0.05,
}: CircularGalleryProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isBrowser, setIsBrowser] = useState(false);
  
  // Set isBrowser to true when component mounts on client-side
  useEffect(() => {
    setIsBrowser(true);
  }, []);
  
  useEffect(() => {
    if (!isBrowser || !containerRef.current) return;
    
    try {
      const app = new App(containerRef.current, {
        items,
        bend,
        textColor,
        backgroundColor,
        borderRadius,
        font,
        scrollSpeed,
        scrollEase,
      });
      return () => {
        app.destroy();
      };
    } catch (error) {
      console.error("Error initializing CircularGallery:", error);
    }
  }, [isBrowser, items, bend, textColor, backgroundColor, borderRadius, font, scrollSpeed, scrollEase]);
  
  return <div className="w-full h-full overflow-hidden cursor-grab active:cursor-grabbing" ref={containerRef} />;
};

export default CircularGallery;
