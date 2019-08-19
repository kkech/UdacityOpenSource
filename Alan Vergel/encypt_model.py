
import face_alignment 
import torch
import syft as sy 

hook = sy.TorchHook(torch)

NUM_WORKERS = 3
workers = [sy.VirtualWorker(hook, id = "w" + str(i)) for i in range(NUM_WORKERS) ]


class EncryptedFA:

    def __init__(self, landmarks_type,
                 device='cuda',
                 flip_input=False, workers = []):

        self.face_alignment = face_alignment.FaceAlignment(landmarks_type = landmarks_type,
                            device = device, flip_input = flip_input )

        # Encrypt model.
        self.face_alignment.face_alignment_net = self.face_alignment.face_alignment_net.fix_precision().share(*workers)
        #Check parameters
        print(next(self.face_alignment.face_alignment_net.parameters()))

fa = EncryptedFA(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)




