cd /root && git clone https://github.com/SC-ProjectDev/vllm_gpu_setup.git && cd vllm_gpu_setup && cp -n .env.example .env && nohup bash bootstrap.sh > /var/log/bootstrap.log 2>&1 &
