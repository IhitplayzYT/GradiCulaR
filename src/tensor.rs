use std::{collections::HashMap, sync::{Arc, LazyLock, Mutex, RwLock}};
use crate::random::Random::Rng;

type NodeId = u64; 
static MAP: LazyLock<RwLock<HashMap<NodeId, Node>>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static RNG: LazyLock<RwLock<Rng>> = LazyLock::new(|| RwLock::new(Rng::seed(0)));

#[derive(Debug,Clone)]
pub struct Node {
    pub value: Vec<f64>,
    pub grad: Option<Vec<f64>>,
    pub op: Option<Op>,
    pub parents: Vec<NodeId>,
    pub requires_grad: bool
}

pub struct Tensor {
    pub id: NodeId,
}

impl Tensor{

    pub fn new(data: &[f64]) -> Self{
        let id = RNG.write().unwrap().next_u64();
        MAP.write().unwrap().insert(id, Node { value: data.to_owned(), grad: None, op: None, parents:vec![], requires_grad: false }); 
        Self {id}
    }

    pub fn requires_grad(&self,requires: bool){
        if let Some(z) = MAP.write().unwrap().get_mut(&self.id){
            z.requires_grad = requires;
        }
    }

    pub fn data(&self) -> Vec<f64>{
        if let Some(z) = MAP.read().unwrap().get(&self.id){
            return z.value.clone();
        }else{
            panic!("Tensor does not exist");
        }
    }

    pub fn grad(&self) -> Option<Vec<f64>>{
        if let Some(z) = MAP.read().unwrap().get(&self.id){
            return z.grad.clone();
        }else{
            panic!("Tensor does not exist");
        }
    }

    pub fn add(&self,other: &Self) -> Self{
        let (lhs, rhs, lhs_requires, rhs_requires) = {
            let map = MAP.read().unwrap();
            let lhs = map.get(&self.id).expect("Left tensor does not exist");
            let rhs = map.get(&other.id).expect("Right tensor does not exist");
            (lhs.value.clone(), rhs.value.clone(), lhs.requires_grad, rhs.requires_grad)
        };

        if lhs.len() != rhs.len(){
            panic!("LHS and RHS need to be equally sized vectors for adding");
        }
        let res = Add.forward(&[&lhs,&rhs]);
        let id = RNG.write().unwrap().next_u64();
        let node = Node {value: res,grad: None,op: Some(Op::Add),parents: vec![self.id,other.id],requires_grad: lhs_requires || rhs_requires};
        MAP.write().unwrap().insert(id, node);
        Tensor { id }
    }

    pub fn backward(&self) {
        // For add dx/dy = [1,...]
        let out_l = MAP.read().unwrap().get(&self.id).expect("Tensor does not exist").value.len();

        MAP.write().unwrap().get_mut(&self.id).expect("Tensor does not exist").grad = Some(vec![1.0;out_l]);

        let (op, parents, grad_output) = {
            let node = MAP.read().unwrap().get(&self.id).expect("Tensor node does not exist").clone();
            (node.op.clone(),node.parents.clone(),node.grad.clone().unwrap())
        };

        let Some(_) = op else { return;};  // Exit if no Operation

        let input_values = {
            let map = MAP.read().unwrap();
            parents.iter().map(|id| {map.get(id).expect("Parent node does not exist").value.clone()}).collect::<Vec<_>>()
        };

        let input_refs = input_values.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let grads = Add.backward(&input_refs, &self.data(), &grad_output);
        let mut map = MAP.write().unwrap();
        for (pid,gd) in parents.iter().zip(grads){
            let parent = map.get_mut(pid).expect("Parent Tensor does not exist");
            if !parent.requires_grad{ continue; }
            match &mut parent.grad{
                Some(existing) => { for (a,b) in existing.iter_mut().zip(gd.iter()){ *a += *b; } },
                _ => { parent.grad = Some(gd); }
            }
        }
    }

}

#[derive(Debug,Clone)]
pub enum Op{
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    Pow(f64)
}

struct Add;

impl AutoGradTensor for Add{

    fn forward(&self,inputs:&[&[f64]]) -> Vec<f64> {
        if inputs.len() != 2{
            panic!("Add requires only two inputs");
        }

        if inputs[0].len() != inputs[1].len(){
            panic!("Can't add unequal length array");
        }
        
        inputs[0].iter().zip(inputs[1]).map(|(a,b)| a+b).collect()
    }
    fn backward(&self,input: &[&[f64]],output: &[f64],grad_output:&[f64]) -> Vec<Vec<f64>> {
        //
        // z = x + y
        // dz/dx = 1
        // dz/dy = 1
        //
        // dL/dx = dL/dz * dz/dx
        // dL/dy = dL/dz * dz/dy
        //

        vec![ grad_output.to_owned(), grad_output.to_owned()]        
    }


}



pub trait AutoGradTensor{
    fn forward(&self,inputs:&[&[f64]]) -> Vec<f64>;
    fn backward(&self,input: &[&[f64]],output: &[f64],grad_output:&[f64]) -> Vec<Vec<f64>>;
}




