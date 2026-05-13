from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from project.train import device, test_loader, models, CHECKPOINT_DIR
import torch

def evaluate_fgsm(model, loader, epsilon):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_images = fgsm_attack(
            model,
            images,
            labels,
            epsilon,
            device
            )

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total

def evaluate_pgd(model, loader, epsilon):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd_attack(
            model,
            images,
            labels,
            epsilon,
            device
            )

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total


print("\nFGSM Evaluation")
epsilon = 0.03
for name, model in models.items():
    model.load_state_dict(
        torch.load(
            f"{CHECKPOINT_DIR}/{name}_best.pth"
        )
    )
    fgsm_acc = evaluate_fgsm(
        model,
        test_loader,
        epsilon
    )
    print(
        f"{name} FGSM Accuracy: "
        f"{fgsm_acc:.2f}%"
    )

print("\n\nPGD Evaluation")
epsilon = 0.03
for name, model in models.items():
    model.load_state_dict(
        torch.load(
            f"{CHECKPOINT_DIR}/{name}_best.pth"
        )
    )
    pgd_acc = evaluate_pgd(
        model,
        test_loader,
        epsilon
    )
    print(
        f"{name} PGD Accuracy: "
        f"{pgd_acc:.2f}%"
    )