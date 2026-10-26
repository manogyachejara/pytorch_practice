import torch
print(torch.__version__)
if torch.cuda.is_available():
  print("gpu is available")
  print(f"using GPU:  {torch.cuda.get_device_name(0)}")

else:
  print("GPU not available")

# creating tensor
#empty
a = torch.empty(2,3)
type(a)

torch.zeros(2,3)

torch.ones(2,3)

torch.rand(2,3)

torch.manual_seed(100)
torch.rand(2,3)

torch.tensor([[1,2,3],[4,5,6]])

print("using arange -> ", torch.arange(0,10,2))

print("using linespace -> ",  torch.linspace(0,10,10))

print("using eye -> ", torch.eye(5))

print("using full -> ", torch.full((3,3),5))

x = torch.tensor([[1,2,3],[4,5,6]])
print(x)
print(x.shape)

print(torch.empty_like(x))

print(torch.zeros_like(x))

print(torch.ones_like(x))

#tensor datatype
torch.rand_like(x, dtype= torch.float32)
print(x.dtype)

torch.tensor([1.0,2.0,3.0], dtype=torch.int32 )
torch.tensor([1,2,3], dtype=torch.float32)
x.to(torch.float32)

# mathematical operations

# scaler operation

x = torch.rand(2,2)
print(x)

print(x + 2)
print(x - 2)
print(x * 2)
print(x / 2)
print((x * 100) // 3)
print(((x * 100) // 3) % 2)
print(x**2)

# element operation

a= torch.rand(2,3)
b= torch.rand(2,3)
print(a)
print(b)

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a%b)

c= torch.tensor([1,-2,3,-4])
print(torch.abs(c))

print(torch.neg(c))

d = torch.rand(2,3)
print(torch.ceil(d))
print(torch.round(d))

print(torch.floor(d))
print(torch.clamp(d, min=2 ,max=3 ))

e = torch.randint(size = (2,3),low =0,high= 10, dtype=torch.float32)
print(e)

print(torch.sum(e))
print(torch.sum(e,dim=0))
print(torch.sum(e,dim=1))
print(torch.mean(e))
print(torch.mean(e,dim=0))
print(torch.median(e))
print(torch.min(e))
print(torch.max(e))
print(torch.std(e))
print(torch.prod(e))
print(torch.var(e))
print(torch.argmax(e))
print(torch.argmin(e))

f = torch.randint(size=(2,2),low=0,high=10)
g = torch.randint(size=(2,2),low=0,high=10)
print(f)
print(g)
print(torch.matmul(f,g))

vector1=torch.tensor([1,2])
vector2=torch.tensor([3,4])
print(torch.dot(vector1,vector2))
print(torch.transpose(f,0,1))

h = torch.rand(3,3, dtype=torch.float32)
print(h)
print(torch.inverse(h))

a = torch.rand(2,3)
print(a)

b = a.clone()

id(a)
id(b)



