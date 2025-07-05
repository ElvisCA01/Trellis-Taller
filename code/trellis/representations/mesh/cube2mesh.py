import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self, vertices, faces, vertex_attrs=None, res=64):
        """
        Contenedor para los resultados de extracción de mallas con propiedades adicionales calculadas.
        
        Args:
                vertices (torch.Tensor): Vértices extraídos [N, 3]
                faces (torch.Tensor): Caras extraídas [M, 3]
                vertex_attrs (torch.Tensor, opcional): Atributos de vértice como colores/normales
                res (int): Resolución de la cuadrícula de extracción
        """
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self._compute_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # Training-only attributes
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def _compute_face_normals(self, verts, faces):
        """Compute per-face normals for the mesh."""
        i0, i1, i2 = faces[..., 0].long(), faces[..., 1].long(), faces[..., 2].long()
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)
        return face_normals[:, None, :].repeat(1, 3, 1)  # Repeat for each vertex in face
                
    def compute_vertex_normals(self, verts, faces):
        """Compute smooth vertex normals by averaging adjacent face normals."""
        i0, i1, i2 = faces[..., 0].long(), faces[..., 1].long(), faces[..., 2].long()
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        vertex_normals = torch.zeros_like(verts)
        
        # Accumulate normals from all adjacent faces
        for idx in [i0, i1, i2]:
            vertex_normals.scatter_add_(0, idx[..., None].repeat(1, 3), face_normals)
            
        return torch.nn.functional.normalize(vertex_normals, dim=-1)


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        """
        Modelo para generar una malla a partir de características dispersas usando FlexiCubes.
        
        Args:
            device (str): Dispositivo de cómputo ('cuda' o 'cpu')
            res (int): Resolución base para la cuadrícula de extracción
            use_color (bool): Si se incluyen atributos de color
        """
        super().__init__()
        self.device = device
        self.res = res
        self.use_color = use_color
        
        # Initialize mesh extractor and regular grid
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res  # Small bias to prevent surface artifacts
        
        # Pre-compute regular grid components
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)  # Regular cube indices
        self.reg_v = verts.to(self.device)  # Regular grid vertices
        
        # Set up feature layout
        self._setup_feature_layouts()
    
    def _setup_feature_layouts(self):
        """Configurar el diseño de memoria para diferentes tipos de características."""
        layouts = {
            'sdf': {'shape': (8, 1), 'size': 8},      # SDF values per cube vertex
            'deform': {'shape': (8, 3), 'size': 24},   # Deformation vectors
            'weights': {'shape': (21,), 'size': 21}    # FlexiCubes weights
        }
        
        if self.use_color:
            layouts['color'] = {'shape': (8, 6), 'size': 48}  # Color + normal info
            
        self.layouts = edict(layouts)
        
        # Calculate memory ranges for each feature type
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
            
        self.feats_channels = start  # Total number of feature channels
    
    def get_layout(self, feats: torch.Tensor, name: str) -> torch.Tensor:
        """Extract features of a specific type from the packed feature tensor."""
        if name not in self.layouts:
            raise ValueError(f"Unknown feature type: {name}")
            
        start, end = self.layouts[name]['range']
        return feats[:, start:end].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats: SparseTensor, training=False) -> MeshExtractResult:
        """
        Genera una malla a partir de características dispersas..
        
        Args:
            cubefeats (SparseTensor): Características dispersas de entrada que contienen:
                - coords: Coordenadas dispersas [N, 4] (lote, x, y, z)
                - feats: Características empaquetadas [N, canales_de_características]
            training (bool): Si está en modo de entrenamiento (calcula pérdidas adicionales)

        Retorna:
            MeshExtractResult: Contiene datos de la malla e información opcional de entrenamiento
        """
        # Unpack coordinates and features
        coords = cubefeats.coords[:, 1:]  # Remove batch dimension
        feats = cubefeats.feats
        
        # Extract different feature types
        sdf = self.get_layout(feats, 'sdf') + self.sdf_bias
        deform = self.get_layout(feats, 'deform')
        weights = self.get_layout(feats, 'weights')
        color = self.get_layout(feats, 'color') if self.use_color else None
        
        # Process vertex attributes
        v_attrs = [sdf, deform]
        if self.use_color:
            v_attrs.append(color)
            
        v_pos, v_attrs, reg_loss = sparse_cube2verts(
            coords, torch.cat(v_attrs, dim=-1), training=training
        )
        
        # Create dense attribute grids
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        
        # Split dense attributes
        sdf_d = v_attrs_d[..., 0]
        deform_d = v_attrs_d[..., 1:4]
        colors_d = v_attrs_d[..., 4:] if self.use_color else None
        
        # Compute deformed vertices
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        # Extract mesh using FlexiCubes
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training
        )
        
        # Package results
        mesh = MeshExtractResult(
            vertices=vertices, 
            faces=faces, 
            vertex_attrs=colors, 
            res=self.res
        )
        
        # Additional training computations
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5  # Add deviation loss
            reg_loss += weights[:, :20].abs().mean() * 0.2  # Weight regularization
            
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
            
        return mesh
