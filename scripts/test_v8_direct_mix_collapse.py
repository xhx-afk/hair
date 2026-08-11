import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def main():
    torch.manual_seed(22)
    high_floor = 0.60
    model = build_adapter(
        direct_mix_init=0.70,
        direct_mix_floor_low=0.10,
        direct_mix_floor_high=high_floor,
    )
    model.set_anchor_trainable(True)
    model.set_correction_trainable(False)
    optimizer = torch.optim.Adam(list(model.anchor_parameters()), lr=5e-2)
    latent_face, latent_color, descriptor = synthetic_inputs(batch=4)
    ones = torch.ones(4)
    zeros = torch.zeros(4)

    initial_output, initial_aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        zeros,
        ones,
        correction_enabled=False,
    )
    del initial_output
    assert torch.allclose(initial_aux["layer_mix"], torch.full_like(initial_aux["layer_mix"], 0.70))

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, aux = run_adapter(
            model,
            latent_face,
            latent_color,
            descriptor,
            ones,
            zeros,
            ones,
            correction_enabled=False,
        )
        aux["layer_mix"].mean().backward()
        optimizer.step()

    _, final_aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        zeros,
        ones,
        correction_enabled=False,
    )
    minimum_mix = float(final_aux["layer_mix"].min())
    assert minimum_mix >= high_floor
    print(f"v8 direct mix collapse test passed: minimum_high_chroma_mix={minimum_mix:.6f}")


if __name__ == "__main__":
    main()
