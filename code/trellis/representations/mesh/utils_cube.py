import torch
# 1. Pre-definir tensores como constantes (evita recrearlos y permite ser compatibles con torch.jit.script)
CUBE_CORNERS = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], 
     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=torch.int32
)  # Usar int32 para compatibilidad con GPUs modernas

CUBE_NEIGHBOR = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], 
     [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.int32
)

CUBE_EDGES = torch.tensor(
    [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
     2, 0, 3, 1, 7, 5, 6, 4],
    dtype=torch.long
)
     
# 2. Optimizar construct_dense_grid con torch.meshgrid (más legible y potencialmente más rápido)
def construct_dense_grid(res, device='cuda'):
    res_v = res + 1
    # Usar meshgrid evita cálculos manuales de índices
    z, y, x = torch.meshgrid(
        torch.arange(res_v, device=device),
        torch.arange(res_v, device=device),
        torch.arange(res_v, device=device),
        indexing='ij'
    )
    verts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    
    # Pre-calcular cube_corners_bias una vez (optimización)
    cube_corners_bias = (CUBE_CORNERS[:, 0] * res_v + CUBE_CORNERS[:, 1]) * res_v + CUBE_CORNERS[:, 2]
    coordsid = verts[:res**3].reshape(-1)  # Más claro que [:res, :res, :res].flatten()
    cube_fx8 = coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device)
    
    return verts, cube_fx8


# 3. construct_voxel_grid con torch.unique con sorted=True (mejor consistencia)
def construct_voxel_grid(coords):
    verts = (CUBE_CORNERS.unsqueeze(0).to(coords) + coords.unsqueeze(1))
    verts = verts.reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(
        verts, dim=0, return_inverse=True, sorted=True
    )
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


# 4. cubes_to_verts con validación de inputs y opción de unsafe (más rápido)
def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    assert cubes.shape[0] == value.shape[0], "Input shapes mismatch"
    assert cubes.shape[1] == 8, "Cubes must have 8 vertices"
    
    M = value.shape[-1]
    reduced = torch.zeros(num_verts, M, device=value.device, dtype=value.dtype)
    
    # Opción 'unsafe' para scatter_reduce si se sabe que no habrá overlaps no deseados
    return torch.scatter_reduce(
        reduced, 0,
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1),
        value.flatten(0, 1),
        reduce=reduce,
        include_self=False
        #, unsafe=True  # Descomentar para rendimiento si es seguro
    )

    
# 5. sparse_cube2verts con modo de entrenamiento desacoplado
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    
    con_loss = 0.0
    if training:
        # Evitar cálculo si no hay gradientes requeridos
        with torch.set_grad_enabled(training):
            con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    
    return new_coords, new_feats, con_loss
    

# 6. get_dense_attrs con indexación más eficiente
def get_dense_attrs(coords: torch.Tensor, feats: torch.Tensor, res: int, sdf_init=True):
    F = feats.shape[-1]
    dense_attrs = torch.zeros((res, res, res, F), device=feats.device, dtype=feats.dtype)
    
    if sdf_init:
        dense_attrs[..., 0].fill_(1)  # Más eficiente que dense_attrs[..., 0] = 1
    
    # Indexación plana para evitar overhead
    dense_attrs.view(-1, F)[
        coords[:, 0] * res * res + coords[:, 1] * res + coords[:, 2]
    ] = feats
    
    return dense_attrs.reshape(-1, F)

# 7. get_defomed_verts con cálculo in-place
def get_defomed_verts(v_pos: torch.Tensor, deform: torch.Tensor, res):
    half_res = (1 - 1e-8) / (res * 2)
    # Usar operaciones in-place donde sea posible
    offset = torch.tanh(deform, out=deform.clone())  # Evitar modificar el tensor original
    offset.mul_(half_res).add_(-0.5)
    
    # Usar torch.div con out para evitar temporal
    v_pos = v_pos.to(dtype=deform.dtype, copy=False)
    return torch.div(v_pos, res, out=v_pos).add_(offset)