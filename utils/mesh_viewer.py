"""Interactive shaded MANO mesh, using predicted vertices and fixed real faces."""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from pathlib import Path
import numpy as np


def vertex_normals(vertices, faces):
    """Area-weighted smooth normals. No geometry is fabricated or fitted here."""
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    result = np.zeros_like(vertices, dtype=np.float32)
    for column in range(3):
        np.add.at(result, faces[:, column], normal)
    lengths = np.linalg.norm(result, axis=1, keepdims=True)
    return np.ascontiguousarray(result / np.maximum(lengths, 1e-10), dtype=np.float32)


def export_obj(path, vertices, faces):
    """Export the actual root-relative predictions, before display normalization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# SimpleHand predicted root-relative hand mesh\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            stream.write(f"f {a+1} {b+1} {c+1}\n")


class MeshViewer:
    def __init__(self, faces, width=720, height=620, hidden=False):
        try:
            import pygame
            from OpenGL import GL, GLU
        except ImportError as exc:
            raise RuntimeError("3D viewer missing libraries. Run: .\\.venv\\Scripts\\python.exe "
                               "-m pip install pygame==2.6.1 PyOpenGL==3.1.7") from exc
        self.pg, self.gl, self.glu = pygame, GL, GLU
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.width, self.height = width, height
        self.yaw, self.pitch, self.distance = -18., -8., 3.4
        self.wireframe, self.running = False, True
        self.reference_extent = None
        self.vertices = self.normals = None
        self.font_cache = {}
        pygame.display.init()
        pygame.font.init()
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            if hidden:
                flags |= pygame.HIDDEN
            pygame.display.set_mode((width, height), flags, vsync=0)
            pygame.display.set_caption("SimpleHand - LIVE 3D HAND MESH - drag to rotate")
            self.font = pygame.font.SysFont("Arial", 19)
            GL.glClearColor(0.035, 0.055, 0.09, 1)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_NORMALIZE)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_LIGHT1)
            GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [.22, .22, .25, 1.])
            GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [.88, .85, .80, 1.])
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, [.30, .40, .55, 1.])
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE, [.28, .72, .82, 1.])
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [.35, .35, .35, 1.])
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 35.)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glDisable(GL.GL_CULL_FACE)  # includes reflected left-hand triangles
            renderer = GL.glGetString(GL.GL_RENDERER)
            self.renderer = renderer.decode(errors="replace") if renderer else "unknown"
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Cannot create OpenGL 3D window: {exc}") from exc

    def update(self, vertices, joints=None):
        if vertices is None:
            self.vertices = self.normals = None
            self.reference_extent = None
            return
        v = np.asarray(vertices, dtype=np.float32)
        if v.shape != (778, 3) or not np.isfinite(v).all():
            raise ValueError("Expected finite SimpleHand vertices (778,3)")
        center = (v.min(0) + v.max(0))*.5
        extent = max(float(np.ptp(v, axis=0).max()), 1e-6)
        if self.reference_extent is None:
            self.reference_extent = extent
        # Uniform scale preserves predicted geometry. First live frame sets zoom scale.
        v = (v-center) * (1.65/self.reference_extent)
        # Camera convention x-right/y-down/z-forward -> OpenGL x-right/y-up/z-back.
        v = np.ascontiguousarray(v * [1., -1., -1.], dtype=np.float32)
        self.vertices = v
        self.normals = vertex_normals(v, self.faces)

    def reset(self):
        self.yaw, self.pitch, self.distance = -18., -8., 3.4
        self.reference_extent = None

    def events(self):
        """Return keyboard actions, so either window can save/quit."""
        pg = self.pg
        actions = set()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    self.running = False
                elif event.key == pg.K_w:
                    self.wireframe = not self.wireframe
                elif event.key == pg.K_r:
                    self.reset()
                elif event.key == pg.K_s:
                    actions.add("save")
            elif event.type == pg.MOUSEMOTION and event.buttons[0]:
                self.yaw += event.rel[0]*.55
                self.pitch = float(np.clip(self.pitch+event.rel[1]*.55, -180, 180))
            elif event.type == pg.MOUSEWHEEL:
                self.distance = float(np.clip(self.distance-event.y*.2, 1.5, 8))
            elif event.type == pg.VIDEORESIZE:
                self.width, self.height = max(event.w, 320), max(event.h, 240)
        if not self.running:
            actions.add("quit")
        return actions

    def _text(self, text, x, y, color=(225, 235, 245), cache=False):
        gl, pg = self.gl, self.pg
        if cache and text in self.font_cache:
            width, height, data = self.font_cache[text]
        else:
            surface = self.font.render(text, True, color)
            width, height = surface.get_size()
            data = pg.image.tostring(surface, "RGBA", True)
            if cache:
                self.font_cache[text] = (width, height, data)
        gl.glRasterPos2f(x, self.height-y-height)
        gl.glDrawPixels(width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)

    def render(self, fps=0., inference_ms=0.):
        gl = self.gl
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self.glu.gluPerspective(42., self.width/self.height, .05, 30.)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [2., 3., 5., 0.])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [-3., -1., 2., 0.])
        gl.glTranslatef(0., -.05, -self.distance)
        gl.glRotatef(self.pitch, 1., 0., 0.)
        gl.glRotatef(self.yaw, 0., 1., 0.)
        if self.vertices is not None:
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertices)
            gl.glNormalPointer(gl.GL_FLOAT, 0, self.normals)
            gl.glDrawElements(gl.GL_TRIANGLES, self.faces.size, gl.GL_UNSIGNED_INT, self.faces)
            if self.wireframe:
                gl.glDisable(gl.GL_LIGHTING)
                gl.glColor3f(.08, .18, .24)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonOffset(-1., -1.)
                gl.glDrawElements(gl.GL_TRIANGLES, self.faces.size, gl.GL_UNSIGNED_INT, self.faces)
                gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, 0, self.height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self._text("SimpleHand - predicted 3D hand surface", 16, 12, cache=True)
        if self.vertices is not None:
            self._text(f"778 vertices / {len(self.faces)} triangles | {fps:.1f} FPS | SH {inference_ms:.1f} ms", 16, 40)
        else:
            self._text("No hand detected - waiting for a live prediction", 16, 40, cache=True)
        self._text("Drag: rotate   Scroll: zoom   W: wireframe   R: reset", 16, self.height-58, cache=True)
        self._text("S: save mesh OBJ + NPZ   Q / Esc: quit", 16, self.height-32, cache=True)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.pg.display.flip()

    def save_screenshot(self, path):
        """Used for render verification, not for creating fake model outputs."""
        import cv2
        if not cv2.imwrite(str(path), self.capture_bgr()):
            raise OSError(f"Cannot save 3D screenshot: {path}")

    def capture_bgr(self):
        """Read only on recording/export, avoiding a GPU readback every live frame."""
        gl = self.gl
        gl.glReadBuffer(gl.GL_FRONT)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.ascontiguousarray(rgb[::-1, :, ::-1])

    def close(self):
        self.running = False
        self.pg.display.quit()
        self.pg.font.quit()
