import numpy as np

def swap(pt1, pt2, start_angle, end_angle, label1, label2, sig, sig2):
    """
    Premiere design d'augmentation des donnees
    Scene-level swapping

    Args:
        pt1, pt2 : Point des deux scans LiDAR
        start_angle, end_angle : Borne pour l'azimute
        label1, label2 : Labels pour les deux scans LiDAR
    """
    # Azimute angle
    theta1 = -np.arctan2(pt1[:,1], pt1[:,0])
    theta2 = -np.arctan2(pt2[:,1], pt2[:,0])

    # Selection des points
    ids1 = np.where((theta1 > start_angle) & (theta1 < end_angle))
    ids2 = np.where((theta2 > start_angle) & (theta2 < end_angle))

    # Cut and delete
    pt1_after = np.delete(pt1, ids1, axis=0)
    la1_after = np.delete(label1, ids1, axis=0)
    pts = np.concatenate((pt1_after, pt2[ids2]))
    labels = np.concatenate((la1_after, label2[ids2]))

    if sig is not None:
        sig1 = np.delete(sig, ids1, axis=0)
        sigout = np.concatenate((sig1, sig2[ids2]), axis=0)
    else:
        sigout = None

    # On s'assure que tout est à la même taille
    assert pts.shape[0] == labels.shape[0]
    return pts, labels, sigout


def rotate(pts, labels, classes, Omega, sig2):
    """
    Deuxieme design d'augmentation des donnees
    Instance level Rotate-pasting 

    Args:
        pt1, pt2 : Point des deux scans LiDAR
        start_angle, end_angle : Borne pour l'azimute
        label1, label2 : Labels pour les deux scans LiDAR
    """
    # Trier les points par classes et instances
    pts_inst, labels_inst = [], []
    if sig2 is not None:
        sig_inst = []

    for s in classes:
        pt_index = np.where((labels==s))
        pts_inst.append(pts[pt_index[0],:])
        labels_inst.append(labels[pt_index[0]])
        if sig2 is not None:
            sig_inst.append(sig2[pt_index[0]])
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)
    if sig2 is not None:
        sig_inst = np.concatenate(sig_inst, axis=0)
        
    # Rotation et copiepour chaque classe
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    if sig2 is not None:
        sig_copy = [sig_inst]

    for omega in Omega:
        mat_rotate = np.array([[np.cos(omega), np.sin(omega), 0],
                               [-np.sin(omega), np.cos(omega), 0],
                               [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:,:3] = np.dot(pts_inst[:,:3], mat_rotate)
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
        if sig2 is not None:
            sig_copy.append(sig_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    if sig2 is not None:
        sig_copy = np.concatenate(sig_copy, axis=0)
    else:
        sig_copy = None
    return pts_copy, labels_copy, sig_copy


def polarmix(sA, yA, sB, yB, C, Omega, alpha, beta, sig, sig2):
    """
    Fonction PolarMix utilisant l'article 
    https://arxiv.org/pdf/2208.00223v1.pdf

    Args :
        {sA, yA}, {sB, yB} : Points et labels de deux scans LiDAR
        C : Liste des classes pour le 'rotate and paste'
        Omega : Liste des angles pour le 'rotate and paste'
        alpha, beta : Borne de l'azimute pour le scene-level swapping
    """
    s, y, sigout = sA, yA, sig
    if np.random.random() < 0.5:
        s, y, sigout = swap(sA, sB, alpha, beta, yA, yB, sig, sig2)
    if np.random.random() < 0.99:
        pts_copy, labels_copy, sig_copy = rotate(sB, yB, C, Omega, sig2)
        s = np.concatenate((s, pts_copy), axis=0)
        y = np.concatenate((y, labels_copy), axis=0)
        if sig2 is not None:
            sigout = np.concatenate((sigout, sig_copy), axis=0)
    return s, y, sigout