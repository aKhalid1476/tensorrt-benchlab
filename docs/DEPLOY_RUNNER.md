# Deploying TensorRT BenchLab Runner

This guide covers deploying the **runner service** on GPU-enabled hosts (AWS, RunPod, Lambda Labs, or local workstation).

## Prerequisites

### Required
- **NVIDIA GPU** with CUDA capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **NVIDIA Drivers** version 525+ (for CUDA 12.1)
- **Docker** with **nvidia-container-toolkit** installed
- **Internet access** for pulling images and downloading model weights

### Recommended GPU Instances
- **AWS EC2**: g4dn.xlarge, g5.xlarge, p3.2xlarge
- **RunPod**: RTX 3090, RTX 4090, A4000, A5000
- **Lambda Labs**: RTX 3090, RTX 4090, A6000
- **Local**: Any modern NVIDIA GPU (GTX 1080+, RTX series, Quadro, Tesla)

## Pre-Deployment: Verify Environment

### 1. Check NVIDIA Drivers

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|...
```

✅ **What to check:**
- Driver version ≥ 525
- CUDA version ≥ 12.0
- GPU is recognized

❌ **If nvidia-smi fails:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-driver-525

# Reboot required
sudo reboot
```

### 2. Verify Docker + NVIDIA Runtime

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Expected:** Should show the same `nvidia-smi` output as above.

❌ **If it fails with "could not select device driver":**

Install nvidia-container-toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Deployment Methods

---

## Method 1: Docker (Recommended)

### Build Runner Image

```bash
# Clone repository
git clone <your-repo-url>
cd tensorrt-benchlab

# Build runner image
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
```

**Build time:** 3-5 minutes (downloads PyTorch, TensorRT, dependencies)

### Run Runner Container

```bash
docker run -d --name runner \
  --gpus all \
  -p 8001:8001 \
  -v $(pwd)/cache:/app/cache \
  -e BENCHLAB_LOG_LEVEL=INFO \
  --restart unless-stopped \
  tensorrt-benchlab-runner
```

**Flags explained:**
- `--gpus all` - Expose all GPUs to container
- `-p 8001:8001` - Expose port 8001
- `-v $(pwd)/cache:/app/cache` - Persist ONNX/TensorRT engine cache
- `--restart unless-stopped` - Auto-restart on failure

### Verify Runner

```bash
# Check container is running
docker ps | grep runner

# Check logs
docker logs -f runner

# Test health endpoint
curl http://localhost:8001/health

# Get version info
curl http://localhost:8001/version | jq .
```

**Expected version response:**
```json
{
  "runner_version": "0.1.0",
  "torch_version": "2.1.0+cu121",
  "cuda_version": "12.1",
  "tensorrt_version": "8.6.1",
  "python_version": "3.11.5",
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "git_commit": "a3f8d9c"
}
```

---

## Method 2: Native Python (Development)

Use this for development or debugging.

### Install Dependencies

```bash
cd tensorrt-benchlab

# Install contracts
cd contracts && pip install -e . && cd ..

# Install runner
cd runner && pip install -e . && cd ..
```

### Run Runner

```bash
cd runner
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

**Logs should show:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     service=runner event=startup version=0.1.0
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## Cloud Provider Guides

### AWS EC2

**1. Launch GPU Instance**
- AMI: Deep Learning AMI (Ubuntu 22.04) - Has drivers pre-installed
- Instance type: g4dn.xlarge (T4 GPU, $0.526/hr) or g5.xlarge (A10G, ~$1/hr)
- Storage: 50 GB (for model cache)
- Security group: Allow inbound TCP 8001

**2. SSH into instance**
```bash
ssh -i your-key.pem ubuntu@<public-ip>
```

**3. Install Docker**
```bash
sudo apt update
sudo apt install -y docker.io nvidia-container-toolkit
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker  # Refresh group membership
```

**4. Deploy runner**
```bash
git clone <repo-url>
cd tensorrt-benchlab
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
docker run -d --name runner --gpus all -p 8001:8001 \
  -v ~/cache:/app/cache tensorrt-benchlab-runner
```

**5. Test**
```bash
curl http://localhost:8001/health
```

**6. Use public IP from controller**
```bash
# On your local machine
curl -X POST http://localhost:8000/runs \
  -d '{"runner_url": "http://<ec2-public-ip>:8001", ...}'
```

**Security Note:** For production, use VPC peering or SSH tunneling instead of exposing port 8001 publicly.

---

### RunPod

**1. Deploy Pod**
- Template: PyTorch (has CUDA drivers)
- GPU: RTX 3090, RTX 4090, or A4000+
- Disk: 50 GB container disk
- Expose TCP port: 8001

**2. SSH into pod**
```bash
ssh root@<pod-ip> -p <pod-ssh-port>
```

**3. Clone and build**
```bash
cd /workspace
git clone <repo-url>
cd tensorrt-benchlab
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
docker run -d --name runner --gpus all -p 8001:8001 \
  -v /workspace/cache:/app/cache tensorrt-benchlab-runner
```

**4. Test**
```bash
curl http://localhost:8001/health
```

**5. Use RunPod's public IP**
- RunPod provides a public IP for each pod
- Use this in controller: `http://<runpod-public-ip>:8001`

---

### Lambda Labs

Similar to RunPod:

**1. Launch instance**
- GPU: RTX 3090, A6000, etc.
- Image: PyTorch (Ubuntu 22.04 with CUDA)

**2. SSH and deploy**
```bash
ssh ubuntu@<lambda-ip>
cd ~
git clone <repo-url>
cd tensorrt-benchlab
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
docker run -d --name runner --gpus all -p 8001:8001 \
  -v ~/cache:/app/cache tensorrt-benchlab-runner
```

---

## Configuration

### Environment Variables

```bash
docker run -d --name runner --gpus all -p 8001:8001 \
  -e BENCHLAB_LOG_LEVEL=DEBUG \
  -e CUDA_VISIBLE_DEVICES=0 \
  tensorrt-benchlab-runner
```

**Available env vars:**
- `BENCHLAB_LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `CUDA_VISIBLE_DEVICES`: GPU IDs to use (default: all)

### Cache Persistence

The runner caches:
- **ONNX models**: `cache/{model}.onnx`
- **TensorRT engines**: `cache/{model}_{precision}_batch{size}_{shape}.trt`
- **Engine metadata**: `cache/{model}_{precision}_batch{size}_{shape}.json`

**Persist cache across container restarts:**
```bash
-v /path/on/host/cache:/app/cache
```

**Clear cache:**
```bash
docker exec runner rm -rf /app/cache/*.onnx /app/cache/*.trt
```

---

## Troubleshooting

### Problem: "RuntimeError: CUDA out of memory"

**Cause:** Batch size too large for GPU VRAM

**Solutions:**
1. Use smaller batch sizes in run request
2. Use a GPU with more VRAM
3. Export `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

### Problem: "Cannot find TensorRT library"

**Cause:** TensorRT not installed in environment

**Solution:**
- Docker image already includes TensorRT
- For native: `pip install tensorrt` (requires CUDA 12.1 compatibility)

### Problem: "Engine build fails with FP16"

**Cause:** GPU doesn't support FP16 (e.g., GTX 1080)

**Solution:**
- Runner automatically falls back to FP32
- Check logs: `precision=fp32` indicates fallback occurred

### Problem: "Sanity check failed - outputs differ"

**Cause:** Quantization error or export issue

**Investigation:**
```bash
# Get detailed run info
curl http://localhost:8000/runs/{run_id} | jq '.environment.sanity_check_passed'
```

**Common reasons:**
- FP16 quantization (expected <0.1% difference)
- ONNX export issue (rebuild cache)

### Problem: Slow first run

**Cause:** Model download + ONNX export + TensorRT engine building

**Expected times (RTX 3090):**
- First run (ResNet50): ~45 seconds (download + export + build)
- Subsequent runs: <2 seconds (cache hit)

**Solution:** Warmup cache before benchmarking:
```bash
curl -X POST http://localhost:8000/runs \
  -d '{"runner_url": "...", "model_name": "resnet50", "engines": ["tensorrt"], "batch_sizes": [1], "num_iterations": 1}'
```

---

## Monitoring

### Check Runner Logs

```bash
docker logs -f runner
```

**Healthy logs:**
```
INFO:     service=runner event=startup version=0.1.0
INFO:     event=nvml_init status=success device_count=1
INFO:     event=telemetry_start sampling_rate=200ms
```

### GPU Utilization

```bash
# Inside container
docker exec runner nvidia-smi

# Or watch continuously
watch -n 1 docker exec runner nvidia-smi
```

### Disk Usage (Cache)

```bash
docker exec runner du -sh /app/cache
```

---

## Security Best Practices

1. **Firewall**: Only allow port 8001 from controller IP
2. **TLS**: Use reverse proxy (nginx) with Let's Encrypt for HTTPS
3. **Auth**: Add API key validation (not implemented yet)
4. **VPC**: Deploy runner in private subnet, use VPC peering

**Example nginx reverse proxy:**
```nginx
server {
    listen 443 ssl;
    server_name runner.example.com;

    ssl_certificate /etc/letsencrypt/live/runner.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/runner.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
    }
}
```

---

## Cost Optimization

### On-Demand vs Spot Instances

**AWS EC2:**
- On-demand g4dn.xlarge: $0.526/hr
- Spot g4dn.xlarge: ~$0.16/hr (70% savings)

**Use spot for development:**
```bash
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --instance-market-options MarketType=spot
```

### Auto-Shutdown

**Stop runner when idle (AWS):**
```bash
# Shutdown after 1 hour of no requests
sudo shutdown -h +60
```

**Lambda Labs:** Instances auto-shutdown when SSH session ends (configure in settings)

---

## Production Checklist

- [ ] NVIDIA drivers installed and verified
- [ ] Docker with nvidia-container-toolkit working
- [ ] Runner container starts successfully
- [ ] Health endpoint responds: `curl http://localhost:8001/health`
- [ ] Version endpoint shows GPU info: `curl http://localhost:8001/version`
- [ ] Cache volume mounted for persistence
- [ ] Firewall rules configured
- [ ] Controller can reach runner URL
- [ ] Test run completes successfully
- [ ] Monitoring/alerting configured (optional)

---

## Next Steps

Once runner is deployed:

1. **Test from controller:**
   ```bash
   curl -X POST http://localhost:8000/runs \
     -H "Content-Type: application/json" \
     -d '{
       "runner_url": "http://<runner-ip>:8001",
       "model_name": "resnet50",
       "engines": ["pytorch_cpu", "pytorch_cuda", "tensorrt"],
       "batch_sizes": [1, 4, 8, 16]
     }'
   ```

2. **Run demo script:**
   ```bash
   RUNNER_URL=http://<runner-ip>:8001 ./scripts/demo.sh
   ```

3. **Review results:**
   ```bash
   curl http://localhost:8000/runs/{run_id} | jq .
   ```

---

## Additional Resources

- [NVIDIA Docker Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [RunPod Documentation](https://docs.runpod.io/)
