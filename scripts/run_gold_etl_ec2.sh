#!/bin/bash
set -euo pipefail

REGION="il-central-1"
INSTANCE_TYPE="r5.4xlarge"  # 16 vCPU, 128 GB RAM
AMI_ID="ami-052dea36aca9f0b4e"  # Amazon Linux 2023
KEY_NAME="winner"
INSTANCE_PROFILE="EC2-Role"
TIMEOUT_MINUTES=60

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_S3_KEY="logs/gold-etl-${TIMESTAMP}.log"

echo "=== Gold ETL - EC2 Runner ==="
echo "Instance: $INSTANCE_TYPE (128 GB RAM)"
echo "Region:   $REGION"
echo ""

# --- Step 1: Ensure default VPC exists ---
echo "[1/5] Checking for default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    echo "  No default VPC found. Creating one..."
    VPC_ID=$(aws ec2 create-default-vpc --region "$REGION" \
        --query 'Vpc.VpcId' --output text)
    echo "  Created default VPC: $VPC_ID"
    CREATED_VPC=true
else
    echo "  Using existing default VPC: $VPC_ID"
    CREATED_VPC=false
fi

# --- Step 2: Write user-data script ---
USERDATA_FILE=$(mktemp)
trap "rm -f $USERDATA_FILE" EXIT

cat > "$USERDATA_FILE" << USERDATA
#!/bin/bash
exec > /var/log/gold-etl.log 2>&1
set -x

echo "=== Gold ETL started at \$(date) ==="

dnf install -y python3.11 python3.11-pip git

cd /tmp
git clone https://github.com/BoazBD/Winner-Scraper.git
cd Winner-Scraper

python3.11 -m pip install -r requirements.txt

echo "=== Running gold_etl.py ==="
python3.11 scripts/gold_etl.py
ETL_EXIT=\$?
echo "=== gold_etl.py exited with code \$ETL_EXIT at \$(date) ==="

aws s3 cp /var/log/gold-etl.log s3://boaz-winner-api/${LOG_S3_KEY} --region ${REGION} || true
shutdown -h now
USERDATA

# --- Step 3: Launch instance ---
echo "[2/5] Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --user-data "file://$USERDATA_FILE" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gold-etl-runner}]' \
    --query 'Instances[0].InstanceId' \
    --output text)
echo "  Instance ID: $INSTANCE_ID"

# --- Step 4: Wait for it to start ---
echo "[3/5] Waiting for instance to enter running state..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
echo "  Instance is running. Gold ETL is executing..."
echo ""

# --- Step 5: Monitor until stopped ---
echo "[4/5] Monitoring instance (timeout: ${TIMEOUT_MINUTES} min)..."
SECONDS_ELAPSED=0
TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))
POLL_INTERVAL=30

while true; do
    STATE=$(aws ec2 describe-instances --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)

    ELAPSED_MIN=$((SECONDS_ELAPSED / 60))
    echo "  [+${ELAPSED_MIN}m] Instance state: $STATE"

    if [ "$STATE" == "stopped" ] || [ "$STATE" == "terminated" ]; then
        echo ""
        echo "  Instance stopped — ETL finished."
        break
    fi

    if [ "$SECONDS_ELAPSED" -ge "$TIMEOUT_SECONDS" ]; then
        echo ""
        echo "  ERROR: Timeout reached (${TIMEOUT_MINUTES} min). Force terminating."
        break
    fi

    sleep $POLL_INTERVAL
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + POLL_INTERVAL))
done

# --- Fetch log ---
echo ""
echo "========== ETL LOG =========="
aws s3 cp "s3://boaz-winner-api/${LOG_S3_KEY}" - --region "$REGION" 2>/dev/null || \
    echo "(Log not yet available in S3. Check s3://boaz-winner-api/${LOG_S3_KEY} later.)"
echo "============================="
echo ""

# --- Step 6: Terminate ---
echo "[5/5] Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
echo "  Instance terminated."
echo ""
echo "=== Done ==="
