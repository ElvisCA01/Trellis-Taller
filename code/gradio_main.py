import os
import sys
sys.path.append(os.getcwd())

try: # Intentar importar xformers, o usar el más rápido (flash-attn) si no están instalados
   import xformers
   os.environ['ATTN_BACKEND'] = 'xformers'
except ImportError:
   os.environ['ATTN_BACKEND'] = 'flash-attn'

os.environ['SPCONV_ALGO'] = 'native' 

import gradio as gr
from gradio_litmodel3d import LitModel3D

import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
import trimesh
import tempfile
import logging
import traceback
import json
import time
from datetime import datetime

from version import code_version

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------- ANÁLISIS DE ARGUMENTOS DE LÍNEA DE COMANDOS -----------------

import argparse
parser = argparse.ArgumentParser(description="Servidor API Trellis")

parser.add_argument("--precision", 
                    choices=["full", "half", "float32", "float16"], 
                    default="full",
                    help="Establece el tamaño de las variables para el pipeline, para ahorrar VRAM y ganar rendimiento")
cmd_args = parser.parse_args()

# ------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

def formatear_tiempo(segundos):
    """Formatear tiempo en formato legible"""
    if segundos < 60:
        return f"{segundos:.2f} segundos"
    elif segundos < 3600:
        minutos = int(segundos // 60)
        segs = segundos % 60
        return f"{minutos}m {segs:.1f}s"
    else:
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segs = segundos % 60
        return f"{horas}h {minutos}m {segs:.1f}s"

def calcular_metricas_basicas(mesh):
    """Calcular solo métricas básicas para evitar bloqueos"""
    metrics = {}
    
    # Métricas simples que no deberían causar problemas
    metrics["conteo_vertices"] = len(mesh.vertices)
    metrics["conteo_caras"] = len(mesh.faces)
    metrics["area"] = float(mesh.area) if hasattr(mesh, 'area') else 0
    
    # Calcular bounding box (caja delimitadora)
    if hasattr(mesh, 'bounding_box') and mesh.bounding_box is not None:
        try:
            metrics["dimensiones_caja_delimitadora"] = mesh.bounding_box.primitive.extents.tolist()
        except:
            metrics["dimensiones_caja_delimitadora"] = None
    
    # Comprobar si es estanco (watertight)
    try:
        metrics["es_estanco"] = bool(mesh.is_watertight)
    except:
        metrics["es_estanco"] = False
    
    # Intentar obtener el volumen si es posible
    if hasattr(mesh, 'is_watertight') and mesh.is_watertight:
        try:
            metrics["volumen"] = float(mesh.volume)
        except:
            metrics["volumen"] = None
    
    return metrics

def calcular_metricas_avanzadas(mesh):
    """Calcular métricas adicionales con protección contra bloqueos"""
    metrics = {}
    
    try:
        # Densidad de vértices
        if hasattr(mesh, 'area') and mesh.area > 0:
            metrics["densidad_vertices"] = len(mesh.vertices) / mesh.area
        else:
            metrics["densidad_vertices"] = 0
            
        # Análisis básico de aristas
        edges = set()
        for face in mesh.faces[:1000]:  # Limitar para evitar bloqueos
            for i in range(3):
                edges.add(tuple(sorted((face[i], face[(i+1)%3]))))
        
        # Contar aristas y calcular la relación aristas/vértices
        metrics["conteo_aristas"] = len(edges)
        metrics["relacion_aristas_vertices"] = len(edges) / len(mesh.vertices) if len(mesh.vertices) > 0 else 0
        
    except Exception as e:
        logger.warning(f"Error calculando métricas avanzadas: {str(e)}")
    
    return metrics

def analizar_modelo_3d(archivo_glb, tiempo_generacion=None, tiempo_extraccion=None):
    """Analizar un archivo GLB y devolver métricas como JSON"""
    if not archivo_glb or not os.path.exists(archivo_glb):
        return "{}"
    
    try:
        logger.info(f"Analizando archivo: {archivo_glb}")
        
        # Cargar la malla
        scene = trimesh.load(archivo_glb)
        
        if scene is None:
            return "{}"
        
        # Extraer la malla de la escena
        mesh = None
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
            if not meshes:
                return "{}"
            mesh = meshes[0]
        else:
            mesh = scene
        
        # Calcular métricas
        metricas = calcular_metricas_basicas(mesh)
        metricas_avanzadas = calcular_metricas_avanzadas(mesh)
        metricas.update(metricas_avanzadas)
        
        # Agregar información de tiempo si está disponible
        if tiempo_generacion is not None:
            metricas["tiempo_generacion_3d"] = formatear_tiempo(tiempo_generacion)
            metricas["tiempo_generacion_3d_segundos"] = round(tiempo_generacion, 2)
        
        if tiempo_extraccion is not None:
            metricas["tiempo_extraccion_glb"] = formatear_tiempo(tiempo_extraccion)
            metricas["tiempo_extraccion_glb_segundos"] = round(tiempo_extraccion, 2)
        
        if tiempo_generacion is not None and tiempo_extraccion is not None:
            tiempo_total = tiempo_generacion + tiempo_extraccion
            metricas["tiempo_total_proceso"] = formatear_tiempo(tiempo_total)
            metricas["tiempo_total_proceso_segundos"] = round(tiempo_total, 2)
        
        # Agregar timestamp del análisis
        metricas["timestamp_analisis"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return json.dumps(metricas, indent=2)
        
    except Exception as e:
        logger.error(f"Error analizando modelo: {str(e)}")
        return "{}"

def iniciar_sesion(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
def finalizar_sesion(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

def preprocesar_imagen(imagen: Image.Image) -> Image.Image:
    """
    Preprocesar la imagen de entrada.
    """
    imagen_procesada = pipeline.preprocess_image(imagen)
    return imagen_procesada

def empaquetar_estado(gs: Gaussian, mesh: MeshExtractResult, tiempo_generacion: float) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
        'tiempo_generacion': tiempo_generacion,
    }
    
def desempaquetar_estado(estado: dict) -> Tuple[Gaussian, edict, float]:
    gs = Gaussian(
        aabb=estado['gaussian']['aabb'],
        sh_degree=estado['gaussian']['sh_degree'],
        mininum_kernel_size=estado['gaussian']['mininum_kernel_size'],
        scaling_bias=estado['gaussian']['scaling_bias'],
        opacity_bias=estado['gaussian']['opacity_bias'],
        scaling_activation=estado['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(estado['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(estado['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(estado['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(estado['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(estado['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(estado['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(estado['mesh']['faces'], device='cuda'),
    )
    
    tiempo_generacion = estado.get('tiempo_generacion', 0)
    
    return gs, mesh, tiempo_generacion

def obtener_semilla(aleatorizar_semilla: bool, semilla: int) -> int:
    """
    Obtener la semilla aleatoria.
    """
    return np.random.randint(0, MAX_SEED) if aleatorizar_semilla else semilla

def imagen_a_3d(
    imagen: Image.Image,
    semilla: int,
    fuerza_guia_ss: float,
    pasos_muestreo_ss: int,
    fuerza_guia_slat: float,
    pasos_muestreo_slat: int,
    req: gr.Request,
) -> Tuple[dict, str, str]:
    """
    Convertir una imagen en un modelo 3D.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    
    # Iniciar medición de tiempo
    tiempo_inicio = time.time()
    logger.info(f"Iniciando generación 3D a las {datetime.now().strftime('%H:%M:%S')}")
    
    outputs = pipeline.run(
        imagen,
        seed=semilla,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": pasos_muestreo_ss,
            "cfg_strength": fuerza_guia_ss,
        },
        slat_sampler_params={
            "steps": pasos_muestreo_slat,
            "cfg_strength": fuerza_guia_slat,
        },
    )
    
    # Calcular tiempo de generación
    tiempo_generacion = time.time() - tiempo_inicio
    logger.info(f"Generación 3D completada en {formatear_tiempo(tiempo_generacion)}")
    
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'muestra.mp4')
    imageio.mimsave(video_path, video, fps=15)
    
    estado = empaquetar_estado(outputs['gaussian'][0], outputs['mesh'][0], tiempo_generacion)
    torch.cuda.empty_cache()
    
    # Crear mensaje de información del tiempo
    info_tiempo = f"✅ Generación completada en: {formatear_tiempo(tiempo_generacion)}"
    
    return estado, video_path, info_tiempo

def extraer_glb(
    estado: dict,
    simplificar_malla: float,
    tamano_textura: int,
    req: gr.Request,
) -> Tuple[str, str, str]:
    """
    Extraer un archivo GLB del modelo 3D.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    
    # Iniciar medición de tiempo para extracción
    tiempo_inicio_extraccion = time.time()
    logger.info(f"Iniciando extracción GLB a las {datetime.now().strftime('%H:%M:%S')}")
    
    gs, mesh, tiempo_generacion = desempaquetar_estado(estado)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=simplificar_malla, texture_size=tamano_textura, verbose=False)
    glb_path = os.path.join(user_dir, 'modelo.glb')
    glb.export(glb_path)
    
    # Calcular tiempo de extracción
    tiempo_extraccion = time.time() - tiempo_inicio_extraccion
    logger.info(f"Extracción GLB completada en {formatear_tiempo(tiempo_extraccion)}")
    
    # Calcular tiempo total
    tiempo_total = tiempo_generacion + tiempo_extraccion
    
    torch.cuda.empty_cache()
    
    # Crear mensaje de información completa del tiempo
    info_tiempo = f"""⏱️ Tiempos de procesamiento:
• Generación 3D: {formatear_tiempo(tiempo_generacion)}
• Extracción GLB: {formatear_tiempo(tiempo_extraccion)}
• Tiempo total: {formatear_tiempo(tiempo_total)}"""
    
    # Guardar los tiempos en el estado global para el análisis
    estado['tiempo_extraccion'] = tiempo_extraccion
    
    return glb_path, glb_path, info_tiempo

def extraer_gaussian(estado: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extraer un archivo Gaussian del modelo 3D.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _, _ = desempaquetar_estado(estado)
    gaussian_path = os.path.join(user_dir, 'modelo.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path

with gr.Blocks(delete_cache=(600, 600), title="Generador 3D con Análisis de Métricas y Tiempo") as demo:
    gr.Markdown("""
    ## Generador de Modelos 3D con Análisis de Métricas y Medición de Tiempo
    Sube una imagen y haz clic en "Generar" para crear un modelo 3D. Se medirá el tiempo de generación y extracción.
    """)
    
    with gr.Row():
        with gr.Column():
            imagen_prompt = gr.Image(label="Imagen de Entrada", format="png", image_mode="RGBA", type="pil", height=300)
            
            with gr.Accordion(label="Configuración de Generación", open=False):
                semilla = gr.Slider(0, MAX_SEED, label="Semilla", value=0, step=1)
                aleatorizar_semilla = gr.Checkbox(label="Aleatorizar Semilla", value=True)
                gr.Markdown("Etapa 1: Generación de Estructura Dispersa")
                with gr.Row():
                    fuerza_guia_ss = gr.Slider(0.0, 10.0, label="Fuerza de Guía", value=7.5, step=0.1)
                    pasos_muestreo_ss = gr.Slider(1, 50, label="Pasos de Muestreo", value=12, step=1)
                gr.Markdown("Etapa 2: Generación de Latente Estructurado")
                with gr.Row():
                    fuerza_guia_slat = gr.Slider(0.0, 10.0, label="Fuerza de Guía", value=3.0, step=0.1)
                    pasos_muestreo_slat = gr.Slider(1, 50, label="Pasos de Muestreo", value=12, step=1)

            boton_generar = gr.Button("Generar Modelo 3D", variant="primary")
            
            # Panel de información de tiempo
            info_tiempo_panel = gr.Textbox(
                label="Información de Tiempo",
                value="⏱️ Los tiempos aparecerán aquí después de generar y extraer",
                interactive=False,
                lines=4
            )
            
            with gr.Accordion(label="Configuración de Extracción GLB", open=False):
                simplificar_malla = gr.Slider(0.9, 0.98, label="Simplificar", value=0.95, step=0.01)
                tamano_textura = gr.Slider(512, 2048, label="Tamaño de Textura", value=1024, step=512)
            
            with gr.Row():
                boton_extraer_glb = gr.Button("Extraer GLB", interactive=False)
                boton_extraer_gs = gr.Button("Extraer Gaussian", interactive=False)
            
            boton_analizar = gr.Button("Analizar Métricas del Modelo", interactive=False, variant="secondary")

        with gr.Column():
            video_salida = gr.Video(label="Modelo 3D Generado", autoplay=True, loop=True, height=300)
            modelo_salida = LitModel3D(label="GLB/Gaussian Extraído", exposure=10.0, height=300)
            
            # Panel de métricas
            metricas_salida = gr.JSON(label="Métricas del Modelo (incluye tiempos)", visible=False)
            
            with gr.Row():
                descarga_glb = gr.DownloadButton(label="Descargar GLB", interactive=False)
                descarga_gs = gr.DownloadButton(label="Descargar Gaussian", interactive=False)  
    
    buffer_salida = gr.State()
    ruta_modelo_actual = gr.State("")

    # Manejadores de eventos
    demo.load(iniciar_sesion)
    demo.unload(finalizar_sesion)
    
    imagen_prompt.upload(
        preprocesar_imagen,
        inputs=[imagen_prompt],
        outputs=[imagen_prompt],
    )

    boton_generar.click(
        obtener_semilla,
        inputs=[aleatorizar_semilla, semilla],
        outputs=[semilla],
    ).then(
        imagen_a_3d,
        inputs=[imagen_prompt, semilla, fuerza_guia_ss, pasos_muestreo_ss, fuerza_guia_slat, pasos_muestreo_slat],
        outputs=[buffer_salida, video_salida, info_tiempo_panel],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[boton_extraer_glb, boton_extraer_gs],
    )

    video_salida.clear(
        lambda: tuple([
            gr.Button(interactive=False), 
            gr.Button(interactive=False), 
            gr.Button(interactive=False), 
            gr.JSON(visible=False),
            gr.Textbox(value="⏱️ Los tiempos aparecerán aquí después de generar y extraer")
        ]),
        outputs=[boton_extraer_glb, boton_extraer_gs, boton_analizar, metricas_salida, info_tiempo_panel],
    )

    boton_extraer_glb.click(
        extraer_glb,
        inputs=[buffer_salida, simplificar_malla, tamano_textura],
        outputs=[modelo_salida, descarga_glb, info_tiempo_panel],
    ).then(
        lambda path: [gr.Button(interactive=True), gr.Button(interactive=True), path],
        inputs=[descarga_glb],
        outputs=[descarga_glb, boton_analizar, ruta_modelo_actual],
    )
    
    boton_extraer_gs.click(
        extraer_gaussian,
        inputs=[buffer_salida],
        outputs=[modelo_salida, descarga_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[descarga_gs],
    )

    boton_analizar.click(
        lambda ruta, estado: [
            json.loads(analizar_modelo_3d(
                ruta, 
                estado.get('tiempo_generacion', None) if estado else None,
                estado.get('tiempo_extraccion', None) if estado else None
            )), 
            gr.JSON(visible=True)
        ],
        inputs=[ruta_modelo_actual, buffer_salida],
        outputs=[metricas_salida, metricas_salida],
    )

    modelo_salida.clear(
        lambda: [gr.Button(interactive=False), gr.JSON(visible=False)],
        outputs=[descarga_glb, metricas_salida],
    )

# Definir una función para inicializar el pipeline
def inicializar_pipeline(precision="full"):
    global pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    # Aplicar configuraciones de precisión. Reducir uso de memoria a costa de precisión numérica:
    print('')
    print(f"Precisión utilizada: '{precision}'. Cargando...")
    print(f"Versión del repositorio Trellis {code_version}")
    if precision == "half" or precision=="float16":
        pipeline.to(torch.float16) # Reduce el uso de memoria a la mitad
        if "image_cond_model" in pipeline.models:
            pipeline.models['image_cond_model'].half()  # Reduce el uso de memoria a la mitad

# Lanzar la aplicación Gradio
if __name__ == "__main__":
    inicializar_pipeline(cmd_args.precision)
    print(f'')
    print(f"Después del lanzamiento, abre un navegador e ingresa 127.0.0.1:7860 (o la IP y puerto que se muestre abajo) en la URL, como si fuera un sitio web:")
    demo.launch()