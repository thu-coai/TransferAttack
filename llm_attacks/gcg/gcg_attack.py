import gc, random
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice, token_num=8):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    # print(f"hessian: {hessian}", flush=True)
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    if token_num < 8:
        target_slice = slice(target_slice.start, target_slice.start+token_num)
    else:
        target_slice = target_slice
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    grad = torch.autograd.grad(loss, one_hot, create_graph=True)[0]
    
    return grad.detach().clone()

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _update_ids(self):
        super()._update_ids()
        target_str = self.target_str
        new_target_str = target_str.split('\n')[0].replace(':', '').strip()
        if "Llama-2" in self.tokenizer.name_or_path:
            target_ids = self.tokenizer.encode("\n" + new_target_str, add_special_tokens=False)
            self.target_in_prompt_slices = self.find_sublist_positions_multiple(target_ids[2:], self.input_ids[:self._target_slice.start])
            target_ids = self.tokenizer.encode("[/INST] " + new_target_str, add_special_tokens=False)
            self.target_res_slice = self.find_sublist_positions_multiple(target_ids[4:], self.input_ids)[-1]
        elif "vicuna" in self.tokenizer.name_or_path:
            target_ids = self.tokenizer.encode('"' + new_target_str, add_special_tokens=False)
            self.target_in_prompt_slices = self.find_sublist_positions_multiple(target_ids[1:], self.input_ids[:self._target_slice.start])
            target_ids = self.tokenizer.encode(new_target_str, add_special_tokens=False)
            self.target_res_slice = self.find_sublist_positions_multiple(target_ids, self.input_ids)[-1]
        if "Llama-3" in self.tokenizer.name_or_path:
            target_ids = self.tokenizer.encode('"' + new_target_str, add_special_tokens=False)
            self.target_in_prompt_slices = self.find_sublist_positions_multiple(target_ids[1:], self.input_ids[:self._target_slice.start])
            target_ids = self.tokenizer.encode("\n\n" + new_target_str, add_special_tokens=False)
            self.target_res_slice = self.find_sublist_positions_multiple(target_ids[1:], self.input_ids)[-1]
          
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice,
            self.token_num,
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True,):

        allow_non_ascii=False
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
            
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def step(self, 
             batch_size=1024,
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1,
             control_weight=0.1,
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)
        
        grad = None
        # Aggregate gradients
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad
                
        with torch.no_grad():
            # breakpoint()
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
    
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                        
                    target_loss = sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    loss[j*batch_size:(j+1)*batch_size] += target_loss
                    if i % 10 == 0:
                        gc.collect()
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
                        
                min_idx = loss.argmin()
                model_idx = min_idx // batch_size
                batch_idx = min_idx % batch_size
                
                next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
                
        del control_cands, loss ; gc.collect()

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
