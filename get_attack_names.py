from attacks.attack_names import AttackNames

for attack_name in AttackNames:
    if attack_name.value != AttackNames.no_attack.value:
        print(attack_name.value)