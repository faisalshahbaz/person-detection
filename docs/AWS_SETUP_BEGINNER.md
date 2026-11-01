# AWS setup for Person Detection — beginner diagrams

Read steps in order. Each section has one picture of **what you create** and **what connects to what**.

Your app: **frontend** (Streamlit), **backend** (FastAPI), **ml-service** (detection).

---

## Step 0 — Big picture (end goal)

```mermaid
flowchart TB
  user[You in browser]
  alb[Application Load Balancer HTTPS]
  fe[Frontend container]
  be[Backend container]
  ml[ML container private]
  redis[(Redis)]
  s3[(S3 videos)]

  user --> alb
  alb --> fe
  alb --> be
  fe --> be
  be --> ml
  be --> redis
  be --> s3
```

Users never talk to ML directly. Path is Frontend → Backend → ML.

---

## Step 1 — One EC2 server (Tier 1 demo, do this first)

**Goal:** Same as docker compose on your laptop, but on AWS.

```mermaid
flowchart LR
  you[Your laptop]
  ssh[SSH port 22]
  ec2[EC2 virtual machine]
  docker[Docker Compose]
  fe2[frontend 8501]
  be2[backend 8000]
  ml2[ml 9000]
  rd2[redis]

  you --> ssh --> ec2
  ec2 --> docker
  docker --> fe2
  docker --> be2
  docker --> ml2
  docker --> rd2
  be2 --> ml2
  be2 --> rd2
```

**Console clicks:** EC2 → Launch instance (t3.small) → Security group: SSH 22 from My IP → SSH in → Docker → docker compose up.

**Webcam via tunnel:**

```mermaid
sequenceDiagram
  participant Browser
  participant SSH as SSH tunnel
  participant EC2 as Streamlit on EC2

  Browser->>SSH: localhost 8501
  SSH->>EC2: port 8501
```

Command: `ssh -L 8501:localhost:8501 user@EC2_IP`

---

## Step 2 — IAM

```mermaid
flowchart TB
  root[Root account billing only]
  iamUser[IAM user deploy-admin]
  cli[AWS CLI keys optional]

  root -->|create| iamUser
  iamUser --> cli
```

IAM → Users → Create user (no daily root login).

---

## Step 3 — VPC

```mermaid
flowchart TB
  subgraph vpc [VPC 10.20.0.0/16]
    pub[Public subnets]
    priv[Private subnets]
  end
  igw[Internet Gateway]
  net[Internet]

  net <--> igw
  igw <--> pub
```

VPC → Create VPC.

---

## Step 4 — Subnets in two zones

```mermaid
flowchart TB
  subgraph azA [Zone A]
    pubA[Public 10.20.0.0/24]
    privA[Private 10.20.8.0/24]
  end
  subgraph azB [Zone B]
    pubB[Public 10.20.1.0/24]
    privB[Private 10.20.9.0/24]
  end
```

Public subnets: auto-assign public IP ON. Private: OFF.

---

## Step 5 — IGW + NAT

```mermaid
flowchart LR
  internet[Internet]
  igw[Internet Gateway]
  pubSub[Public subnet]
  nat[NAT Gateway]
  privSub[Private subnet]
  tasks[Fargate tasks]

  internet <--> igw <--> pubSub
  pubSub --> nat --> privSub
  tasks --> privSub
```

Private tasks pull ECR images through NAT (outbound only).

---

## Step 6 — Security groups

```mermaid
flowchart LR
  alb[sg-alb 443 from world]
  fe[sg-frontend from alb]
  be[sg-backend from alb and fe]
  ml[sg-ml from backend only]
  redis[sg-redis from backend only]

  alb --> fe
  alb --> be
  fe --> be
  be --> ml
  be --> redis
```

---

## Step 7 — ECR images

```mermaid
flowchart LR
  pc[Your PC docker build]
  ecr[Amazon ECR]
  ecs[ECS Fargate pull]

  pc -->|docker push| ecr
  ecs -->|pull| ecr
```

Three repos: frontend, backend, ml-service.

---

## Step 8 — Secrets + S3

```mermaid
flowchart TB
  sec[Secrets Manager JWT ML token]
  s3[S3 bucket uploads]
  be[Backend task]

  sec --> be
  be --> s3
```

---

## Step 9 — Redis

```mermaid
flowchart LR
  be[Backend]
  redis[(ElastiCache Redis)]
  be --> redis
```

Private subnets + sg-redis.

---

## Step 10 — ACM + ALB (HTTPS)

```mermaid
flowchart TB
  user[Browser]
  acm[ACM certificate]
  alb[Application Load Balancer]
  tgFe[Target group frontend 8501]
  tgBe[Target group backend 8000]

  acm --> alb
  user --> alb
  alb --> tgFe
  alb -->|host api.domain| tgBe
```

---

## Step 11 — ECS Fargate

```mermaid
flowchart TB
  cluster[ECS Cluster]
  svcFe[Service frontend]
  svcBe[Service backend]
  svcMl[Service ml private]

  cluster --> svcFe
  cluster --> svcBe
  cluster --> svcMl
```

Frontend and backend register to ALB. ML has no public URL.

```mermaid
flowchart LR
  alb[ALB]
  fe[frontend task]
  be[backend task]
  ml[ml task]

  alb --> fe
  alb --> be
  be --> ml
```

---

## Step 12 — Cloud Map (ML DNS)

```mermaid
flowchart LR
  be[Backend]
  dns[ml.person-detection.local]
  ml[ML task]

  be --> dns --> ml
```

---

## Step 13 — Route 53

```mermaid
flowchart LR
  user[Browser]
  r53[Route 53]
  alb[ALB]

  user -->|app and api domains| r53 --> alb
```

Set frontend BACKEND_URL=https://api.yourdomain.com

---

## Step 14 — WebRTC NLB optional

```mermaid
flowchart LR
  browser[Browser UDP]
  nlb[Network Load Balancer]
  turn[coturn]
  fe[Frontend]

  browser --> nlb --> turn
  turn -.-> fe
```

Skip until public domain webcam is required.

---

## Step 15 — Request path check

```mermaid
sequenceDiagram
  participant U as Browser
  participant FE as Frontend
  participant BE as Backend
  participant ML as ML
  participant R as Redis

  U->>FE: UI and WebRTC
  FE->>BE: JWT login infer
  BE->>R: sessions
  BE->>ML: predict token
  ML-->>BE: boxes
  BE-->>FE: results
```

---

## Beginner order (table)

| Order | What | Section |
|------:|------|---------|
| 1 | EC2 + compose | Step 1 |
| 2 | IAM | Step 2 |
| 3 | VPC + subnets | 3–4 |
| 4 | IGW + NAT | 5 |
| 5 | Security groups | 6 |
| 6 | ECR | 7 |
| 7 | Secrets + S3 | 8 |
| 8 | Redis | 9 |
| 9 | ACM + ALB | 10 |
| 10 | ECS + Cloud Map | 11–12 |
| 11 | Route 53 | 13 |
| 12 | NLB WebRTC | 14 |

Checklist text: AWS_MANUAL_SETUP.txt

## Local first (same idea as Step 1)

```mermaid
flowchart LR
  browser[localhost 8501]
  fe[frontend]
  be[backend]
  ml[ml-service]
  redis[redis]

  browser --> fe --> be --> ml
  be --> redis
```
