#!/bin/bash
# tmp.csv should be a CSV with two columns: instance id and public DNS (IP)

# https://unix.stackexchange.com/questions/48425/how-to-stop-the-loop-bash-script-in-terminal/48465
trap "exit" INT
while IFS=, read -r col1 col2
do
    instanceid=$col1
    ip=$col2
    scp -i /home/vitchyr/git/doodad/aws_config/private/key_pairs/doodad-us-west-1.pem \
        -oStrictHostKeyChecking=no \
        ubuntu@$ip:/tmp/doodad-output/variant.json /tmp/variant.json\
        > /dev/null
    # This kills all instance where the version is equal to "DDPG-TDM"
    value=$(cat /tmp/variant.json | jq .env_class | jq .["$class"])
    echo $value
    if [ $value = '"railrl.envs.multitask.ant_env.GoalXYPosAndVelAnt"' ]; then
        echo aws ec2 terminate-instances --instance-ids $instanceid
        aws ec2 terminate-instances --instance-ids $instanceid
#        echo $instanceid
    else
        echo OKAY
    fi
done < $1
