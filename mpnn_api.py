import argparse
import sys
import copy
from prody import writePDB, writePDBStream
import io
import torch
import random
import json
import numpy as np
import pandas as pd
import os.path
import data_utils
from data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from sc_utils import Packer, pack_side_chains
import time



class MPNNRunner(object):
    
    def __init__(self, model_type, checkpoint_path=None, ligand_mpnn_use_side_chain_context=False,
                 pack_sc=False, pack_sc_checkpoint_path=None, seed=None, verbose=False):
        """
        Creates an instance of MPNN sequence design model runner.
        The workflow logic is:
        1) Create instance of runner: runner = MPNNRunner(*args)
        2) Create mpnn input object: inp = runner.MPNN_Input()
        3) 
            model_type (str) :: must be one of:
                ['protein_mpnn', 'ligand_mpnn', 'per_residue_label_membrane_mpnn',
                 'global_label_membrane_mpnn', 'soluble_mpnn']
            ligand_mpnn_use_side_chain_context (bool)
            pack_sc (bool)
            pack_sc_checkpoint_path (str)
            seed (int)
            verbose (bool)
        """

        #fix seeds
        if seed is not None:
            self.seed=seed
        else:
            self.seed=int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Assuming that by default the user has downloaded the weights into LigandMPNN directory
        # following the instructions in the repo.
        SCRIPT_DIR = os.path.dirname(__file__)
        __checkpoints = {"protein_mpnn": f"{SCRIPT_DIR}/model_params/proteinmpnn_v_48_020.pt",
                         "ligand_mpnn": f"{SCRIPT_DIR}/model_params/ligandmpnn_v_32_020_25.pt",
                         "per_residue_label_membrane_mpnn": f"{SCRIPT_DIR}/model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
                         "global_label_membrane_mpnn": f"{SCRIPT_DIR}/model_params/global_label_membrane_mpnn_v_48_020.pt",
                         "soluble_mpnn": f"{SCRIPT_DIR}/model_params/solublempnn_v_48_020.pt"}

        assert model_type in __checkpoints.keys(), "invalid model_type input"

        if checkpoint_path is None:
            self.__checkpoint_path = __checkpoints[model_type]
        else:
            # TODO: some sanity check for whether the user provides the correct checkpoint?
            self.__checkpoint_path = checkpoint_path

        self.__model_type = model_type

        self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context

        print(f"Using {model_type} model from checkpoint: {self.__checkpoint_path}")

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        checkpoint = torch.load(self.__checkpoint_path, map_location=self.device)


        if self.__model_type == "ligand_mpnn":
            self.atom_context_num = checkpoint["atom_context_num"]
            # self.atom_context_num = 25  # TODO: load from weights
            # k_neighbors=32
            k_neighbors = checkpoint["num_edges"]
        else:
            self.atom_context_num = 1
            ligand_mpnn_use_side_chain_context = 0
            k_neighbors=checkpoint["num_edges"]

        self.model = ProteinMPNN(node_features=128,
                        edge_features=128,
                        hidden_dim=128,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        k_neighbors=k_neighbors,
                        device=self.device,
                        atom_context_num=self.atom_context_num,
                        model_type=self.__model_type,
                        ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        # missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False) # enhanced_MPNN
        # assert len(missing_keys) == 0, f"Missing keys: {missing_keys}" # enhanced_MPNN
        self.model.to(self.device)
        self.model.eval()


        #load side chain packing model if needed
        self.checkpoint_sc = None
        self.model_sc = None
        if pack_sc:
            self.model_sc = Packer(node_features=128,
                            edge_features=128,
                            num_positional_embeddings=16,
                            num_chain_embeddings=16,
                            num_rbf=16,
                            hidden_dim=128,
                            num_encoder_layers=3,
                            num_decoder_layers=3,
                            atom_context_num=16,
                            lower_bound=0.0,
                            upper_bound=20.0,
                            top_k=32,
                            dropout=0.0,
                            augment_eps=0.0,
                            atom37_order=False,
                            device=self.device,
                            num_mix=3)
            if pack_sc_checkpoint_path is None:
                pack_sc_checkpoint_path = "/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt"
            self.checkpoint_sc = torch.load(pack_sc_checkpoint_path, map_location=self.device)
            self.model_sc.load_state_dict(self.checkpoint_sc['model_state_dict'])
            self.model_sc.to(self.device)
            self.model_sc.eval()

        self.verbose = verbose
        pass

    def checkpoint(self):
        return self.__checkpoint

    @property
    def model_type(self):
        return self.__model_type

    class MPNN_Input(object):
        def __init__(self, obj=None):
            ## This just crudely replicates the args object with its attribute namespace

            self.temperature = None
            self.batch_size = None
            self.number_of_batches = None

            self.name = None
            self.max_length = 20000

            self.pdb = None
            self.fixed_residues = []  # ["A1", "A100"]
            self.design_residues = []  # ["A1", "A100"]
            self.fix_everything_not_designable = False
            self.parse_these_chains_only = None
            self.chains_to_design = None
            self.repack_everything = True
            self.parse_atoms_with_zero_occupancy = True
            self.fasta_seq_separation = "/"  # Symbol to use between sequences from different chains

            self.omit_AA = []
            self.omit_AA_per_residue = None  # {"A1": ["C", "M"], ...}
            self.bias_AA = None  # dict of {aa1: bias}
            self.bias_AA_per_residue = None
            self.transmembrane_buried = None
            self.transmembrane_interface = None
            self.global_transmembrane_label = None

            self.symmetry_residues = None
            self.symmetry_weights = None  # 1.0 is applied to all positions if this is not defined, but symmetry_residues is defined
            self.homo_oligomer = False
            self.verbose = None
            self.ligand_mpnn_cutoff_for_score = 8.0
            self.ligand_mpnn_use_atom_context = True

            self.number_of_packs_per_design = 1
            self.sc_num_denoising_steps = 3
            self.sc_num_samples = 16
            self.repack_everything = None
            self.zero_indexed = None
            self.force_hetatm = None

            # Cloning the values of input object, or reading from dictionary keys and values
            if obj is not None:
                if isinstance(obj, MPNNRunner.MPNN_Input):
                    for attr in obj.__dir__():
                        if attr[:2] == "__":
                            continue
                        self.__setattr__(attr, copy.deepcopy(obj.__getattribute__(attr)))
                elif isinstance(obj, dict):
                    self.create_from_dict(obj)
            pass

        def copy(self):
            return copy.deepcopy(self)
        
        def create_from_dict(self, dct):
            for k,v in dct.items():
                self.__setattr__(k, copy.deepcopy(v))
            pass


    def run(self, input_obj, use_sc=True, use_DNA_RNA=True, use_ligand=True, pack_sc=False, num_packs=1, return_pdb=False, **kwargs):
        """
        Runs the MPNN sequence design model on input provided through the `input_obj` object.
        """
        if pack_sc:
            if self.model_sc is None:
                print("Cannot run sidechain packer, please create an instance of MPNNRunner with 'pack_sc=True' and try again.")
                return None

        if self.verbose:
            print("Designing this PDB:", input_obj.name)

        fixed_residues = input_obj.fixed_residues
        redesigned_residues = input_obj.design_residues

        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        if input_obj.bias_AA:
            assert isinstance(input_obj.bias_AA, dict)
            for AA, bias in input_obj.bias_AA.items():
                bias_AA[restype_str_to_int[AA]] = bias

        #make array to indicate which amino acids need to be omitted [21]
        omit_AA = torch.tensor(np.array([AA in input_obj.omit_AA for AA in alphabet]).astype(np.float32), device=self.device)

        #parse PDB file
        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(input_obj.pdb,
                                                                            device=self.device,
                                                                            chains=input_obj.parse_these_chains_only,
                                                                            parse_all_atoms=self.ligand_mpnn_use_side_chain_context or not input_obj.repack_everything,
                                                                            parse_atoms_with_zero_occupancy=input_obj.parse_atoms_with_zero_occupancy)

        #----
        #make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))

        bias_AA_per_residue = torch.zeros([len(encoded_residues),21], device=self.device, dtype=torch.float32)
        if input_obj.bias_AA_per_residue:  # format {chain}{resno}
            if len(input_obj.bias_AA_per_residue) !=0 and len([k for k in input_obj.bias_AA_per_residue if k in encoded_residues]) == 0:
                sys.exit("bias_AA_per_residue dictionary was provided, but none of the keys match the residues in encoded_residues.\nCheck your input...")
            for k in input_obj.bias_AA_per_residue.keys():
                assert isinstance(k, str)
                assert not k[0].isnumeric()
            bias_dict = input_obj.bias_AA_per_residue
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1,j1] = v2
        #----

        omit_AA_per_residue = torch.zeros([len(encoded_residues),21], device=self.device, dtype=torch.float32)
        if input_obj.omit_AA_per_residue:    
            omit_dict = input_obj.omit_AA_per_residue
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1,j1] = 1.0
        #----
        
        if len(fixed_residues) == 0 and len(redesigned_residues) != 0 and input_obj.fix_everything_not_designable is True:
            fixed_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=self.device)
        else:
            fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=self.device)
        redesigned_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=self.device)
        #----

        #specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if input_obj.transmembrane_buried:
            buried_residues = input_obj.transmembrane_buried
            buried_positions = torch.tensor([int(item in buried_residues) for item in encoded_residues], device=self.device)
        else:
            buried_positions = torch.zeros_like(fixed_positions)
        #----

        if input_obj.transmembrane_interface:
            interface_residues = input_obj.transmembrane_interface
            interface_positions = torch.tensor([int(item in interface_residues) for item in encoded_residues], device=self.device)
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        #----
        protein_dict["membrane_per_residue_labels"] = 2*buried_positions*(1-interface_positions) + 1*interface_positions*(1-buried_positions)

        if self.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = input_obj.global_transmembrane_label + 0*fixed_positions
        

        #specify which chains need to be redesigned
        if isinstance(input_obj.chains_to_design, str):
            chains_to_design_list = input_obj.chains_to_design.split(",")
        elif isinstance(input_obj.chains_to_design, list):
            chains_to_design_list = input_obj.chains_to_design
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(np.array([item in chains_to_design_list for item in protein_dict["chain_letters"]],dtype=np.int32), device=self.device)
        #----

        #create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if self.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)
        #----


        #----
        #specify which residues are linked
        # linked positions are items in one list. Each separate position is a separate list.
        # Below, A1, B1 and C1 are tied together.
        # [[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]]
        if input_obj.symmetry_residues:
            symmetry_residues_list_of_lists = input_obj.symmetry_residues
            remapped_symmetry_residues=[]
            for t_list in symmetry_residues_list_of_lists:
                tmp_list=[]
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list) 
        else:
            remapped_symmetry_residues=[[]]
        #----

        #specify linking weights
        if input_obj.symmetry_weights:
            symmetry_weights = [[float(item) for item in x.split(',')] for x in input_obj.symmetry_weights.split('|')]
        else:
            symmetry_weights = [[]]
            if input_obj.symmetry_residues:
                # 1.0 symmetry weights to all symmetry_residues if user did not provide a weight
                symmetry_weights = [[1.0]*len(x) for x in input_obj.symmetry_residues]
        #----

        if input_obj.homo_oligomer:
            if input_obj.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            symmetry_weights = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        #set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors*0.0)
        #----

        #adjust input PDB name by dropping .pdb if it does exist
        name = input_obj.name
        if name is not None and name[-4:] == ".pdb":
            name = name[:-4]
        #----

        out_dict = {}

        with torch.no_grad():
            #run featurize to remap R_idx and add batch dimension
            feature_dict = featurize(protein_dict,
                                    cutoff_for_score=input_obj.ligand_mpnn_cutoff_for_score, 
                                    use_atom_context=input_obj.ligand_mpnn_use_atom_context,
                                    number_of_ligand_atoms=self.atom_context_num,
                                    model_type=self.model_type)
            feature_dict["batch_size"] = input_obj.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            #----

            #add additional keys to the feature dictionary
            feature_dict["temperature"] = input_obj.temperature
            feature_dict["bias"] = (-1e8*omit_AA[None,None,:]+bias_AA).repeat([1,L,1])+bias_AA_per_residue[None]-1e8*omit_AA_per_residue[None]
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            #----

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_per_residue_list_UM = []
            loss_XY_list = []
            for _ in range(input_obj.number_of_batches):
                feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=self.device)
                #main step-----
                output_dict = self.model.sample(feature_dict)

                #compute confidence scores
                loss, loss_per_residue = get_score(output_dict["S"], output_dict["log_probs"], feature_dict["mask"]*feature_dict["chain_mask"])
                loss_unmasked, loss_per_residue_unmasked = get_score(output_dict["S"], output_dict["log_probs"], feature_dict["mask"])
                if self.model_type == "ligand_mpnn":
                    combined_mask = feature_dict["mask"]*feature_dict["mask_XY"]*feature_dict["chain_mask"]
                else:
                    combined_mask = feature_dict["mask"]*feature_dict["chain_mask"]
                loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask)
                #-----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_per_residue_list_UM.append(loss_per_residue_unmasked)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_per_residue_stack_unmasked = torch.cat(loss_per_residue_list_UM, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)


            #side chain packing
            #---------------
            #---------------
            if pack_sc:
                if self.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=input_obj.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                # out_dict["packed"] = {}
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = input_obj.batch_size
                for k,v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B,1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B,1,1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B,1,1,1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B,1,1,1,1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(input_obj.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(input_obj.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(sc_feature_dict, self.model_sc,
                                                   input_obj.sc_num_denoising_steps,
                                                   input_obj.sc_num_samples,
                                                   input_obj.repack_everything)
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            #---------------
            #---------------
            
            #make input sequence string separated by / between different chains
            native_seq = "".join([restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()])
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict['mask_c']:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [input_obj.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]
            native_seq_str = seq_out_str
            #------

            out_dict["generated_sequences_int"] = S_stack.detach().cpu().numpy()
            out_dict["generated_sequences"] = []
            for ix in range(S_stack.shape[0]):
                seq = "".join(
                    [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                )
                seq_np = np.array(list(seq))
                seq_out_str = []
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [input_obj.fasta_seq_separation]
                out_dict["generated_sequences"].append("".join(seq_out_str)[:-1])
            # out_dict["generated_sequences"] = ["".join([restype_int_to_str[AA] for AA in s_ix]) for s_ix in S_stack.detach().cpu().numpy()]
            out_dict["scores"] = [np.exp(-L_ix) for L_ix in loss_stack.detach().cpu().numpy()]
            out_dict["scores_per_residue"] = [np.exp(-L_ix) for L_ix in loss_per_residue_stack.detach().cpu().numpy()]
            # out_dict["scores_per_residue_unmasked"] = [np.exp(-L_ix) for L_ix in loss_per_residue_stack_unmasked.detach().cpu().numpy()]
            out_dict["sampling_probs"] = sampling_probs_stack.detach().cpu().numpy()
            out_dict["sampling_probs_dict"] = [[{restype_int_to_str[i]: v for i,v in enumerate(res_probs)} for res_probs in seqprobs] for seqprobs in out_dict["sampling_probs"]]
            out_dict["log_probs"] = log_probs_stack.detach().cpu().numpy()
            out_dict["decoding_order"] = decoding_order_stack.detach().cpu().numpy()
            out_dict["native_sequence_int"] = feature_dict["S"][0].detach().cpu().numpy()
            out_dict["native_sequence"] = native_seq_str
            out_dict["mask"] = feature_dict["mask"][0].detach().cpu().numpy()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].detach().cpu().numpy()
            out_dict["seed"] = self.seed
            out_dict["temperature"] = input_obj.temperature
            out_dict["packed"] = None
            out_dict["pdb"] = None

            if pack_sc:
                out_dict["packed"] = {}
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not input_obj.zero_indexed:
                        ix_suffix += 1
    
                    #write full PDB files
                    out_dict["packed"][ix] = []
                    for c_pack in range(input_obj.number_of_packs_per_design):
                        X_stack = X_stack_list[c_pack]
                        X_m_stack = X_m_stack_list[c_pack]
                        b_factor_stack = b_factor_stack_list[c_pack]
                        _io_out = io.StringIO()  # memory object where PDB string is saved
                        write_full_PDB(_io_out, X_stack[ix].cpu().numpy(),
                            X_m_stack[ix].cpu().numpy(),
                            b_factor_stack[ix].cpu().numpy(),
                            feature_dict["R_idx_original"][0].cpu().numpy(),
                            protein_dict["chain_letters"], S_stack[ix].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                            force_hetatm=input_obj.force_hetatm)
                        out_dict["packed"][ix].append(_io_out.getvalue())
                        _io_out.close()
                    #-----

            if return_pdb:
                # Just a PDBstring with backbone atoms assigned to designed residue names
                out_dict["pdb"] = []
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not input_obj.zero_indexed:
                        ix_suffix += 1
                    seq = "".join(
                        [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                    )

                    # write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                        None,
                    ].repeat(4, 1)
                    bfactor_prody = (
                        loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                    )
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(
                        np.exp(-bfactor_prody)
                        * (bfactor_prody > 0.01).astype(np.float32)
                    )
                    _io_out = io.StringIO()  # memory object where PDB string is saved
                    if other_atoms:
                        writePDBStream(_io_out, backbone + other_atoms)
                    else:
                        writePDBStream(_io_out, backbone)
                    out_dict["pdb"].append(_io_out.getvalue())
                    _io_out.close()

            return out_dict
