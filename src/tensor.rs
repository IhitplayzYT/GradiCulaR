use std::{collections::HashMap, sync::{Arc, LazyLock, Mutex, RwLock}};
use crate::random::Random::Rng;

type NodeId = u64; 

static MAP: LazyLock<RwLock<HashMap<NodeId, Node>>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static RNG: LazyLock<RwLock<Rng>> = LazyLock::new(|| RwLock::new(Rng::seed(0)));

pub struct Graph {
    nodes: Vec<Node>,
}

#[derive(Debug,Clone)]
pub struct Node {
    pub value: Vec<f64>,
    pub grad: Option<Vec<f64>>,
    pub op: Option<Op>,
    pub parents: Vec<NodeId>,
    pub requires_grad: bool
}

pub struct Tensor {
    id: NodeId,
    graph: Arc<Mutex<Graph>>,
}

impl Tensor{

    pub fn new(data: &[f64]) -> Self{
        let id = RNG.write().unwrap().next_u64();
        let nd = Node { value: data.to_owned(), grad: None, op: None, parents:vec![], requires_grad: false };
        MAP.write().unwrap().insert(id,nd.clone()); 
        Self { id, graph: Arc::new(Mutex::new(Graph { nodes: vec![nd] })) }
    }

    pub fn requires_grad(&self,requires: bool){
        if let Some(z) = MAP.write().unwrap().get_mut(&self.id){
            z.requires_grad = requires;
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

pub trait AutoGradTensor{
    fn forward(&self,inputs:&[&[f64]]) -> Vec<f64>;
    fn backward(&self,input: &[&[f64]],output: &[f64],grad_output:&[f64]) -> Vec<Vec<f64>>;
}




