import numpy as np

from tinynav.core.vlad_retrieval import compute_vlad, fit_vlad_codebook, train_vocabulary_streaming


def test_fit_vlad_codebook_shape():
    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(500, 16)).astype(np.float32)
    codebook = fit_vlad_codebook(descriptors, num_clusters=8, num_iters=10)
    assert codebook.shape == (8, 16)


def test_compute_vlad_output_dimension_and_norm():
    rng = np.random.default_rng(1)
    descriptors = rng.normal(size=(256, 32)).astype(np.float32)
    codebook = fit_vlad_codebook(descriptors, num_clusters=4, num_iters=10)
    vlad = compute_vlad(descriptors, codebook)
    assert vlad.shape == (4 * 32,)
    np.testing.assert_allclose(np.linalg.norm(vlad), 1.0, atol=1e-5)


def test_compute_vlad_identical_input_is_identical_output():
    rng = np.random.default_rng(2)
    descriptors = rng.normal(size=(128, 8)).astype(np.float32)
    codebook = fit_vlad_codebook(descriptors, num_clusters=4, num_iters=10)
    vlad_a = compute_vlad(descriptors, codebook)
    vlad_b = compute_vlad(descriptors, codebook)
    np.testing.assert_allclose(vlad_a, vlad_b)


def test_compute_vlad_discriminates_different_scenes():
    rng = np.random.default_rng(3)
    codebook = fit_vlad_codebook(rng.normal(size=(500, 16)).astype(np.float32), num_clusters=8, num_iters=10)

    descriptors_a = rng.normal(loc=0.0, size=(200, 16)).astype(np.float32)
    descriptors_a_repeat = descriptors_a + rng.normal(scale=0.01, size=descriptors_a.shape).astype(np.float32)
    descriptors_b = rng.normal(loc=5.0, size=(200, 16)).astype(np.float32)

    vlad_a = compute_vlad(descriptors_a, codebook)
    vlad_a_repeat = compute_vlad(descriptors_a_repeat, codebook)
    vlad_b = compute_vlad(descriptors_b, codebook)

    similarity_same_scene = float(vlad_a @ vlad_a_repeat)
    similarity_diff_scene = float(vlad_a @ vlad_b)
    assert similarity_same_scene > similarity_diff_scene


def test_train_vocabulary_streaming_shape():
    rng = np.random.default_rng(4)
    frames = [rng.normal(size=(64, 16)).astype(np.float32) for _ in range(20)]
    codebook = train_vocabulary_streaming(lambda: iter(frames), num_clusters=8, batch_size=128)
    assert codebook.shape == (8, 16)


def test_train_vocabulary_streaming_matches_batch_size_that_does_not_divide_evenly():
    rng = np.random.default_rng(5)
    frames = [rng.normal(size=(50, 16)).astype(np.float32) for _ in range(7)]
    codebook = train_vocabulary_streaming(lambda: iter(frames), num_clusters=4, batch_size=128)
    assert codebook.shape == (4, 16)
    assert np.isfinite(codebook).all()


def test_train_vocabulary_streaming_discriminates_different_scenes():
    rng = np.random.default_rng(6)
    frames = [rng.normal(size=(64, 16)).astype(np.float32) for _ in range(20)]
    codebook = train_vocabulary_streaming(lambda: iter(frames), num_clusters=8, batch_size=128)

    descriptors_a = rng.normal(loc=0.0, size=(200, 16)).astype(np.float32)
    descriptors_a_repeat = descriptors_a + rng.normal(scale=0.01, size=descriptors_a.shape).astype(np.float32)
    descriptors_b = rng.normal(loc=5.0, size=(200, 16)).astype(np.float32)

    vlad_a = compute_vlad(descriptors_a, codebook)
    vlad_a_repeat = compute_vlad(descriptors_a_repeat, codebook)
    vlad_b = compute_vlad(descriptors_b, codebook)

    similarity_same_scene = float(vlad_a @ vlad_a_repeat)
    similarity_diff_scene = float(vlad_a @ vlad_b)
    assert similarity_same_scene > similarity_diff_scene


if __name__ == "__main__":
    test_fit_vlad_codebook_shape()
    test_compute_vlad_output_dimension_and_norm()
    test_compute_vlad_identical_input_is_identical_output()
    test_compute_vlad_discriminates_different_scenes()
    test_train_vocabulary_streaming_shape()
    test_train_vocabulary_streaming_matches_batch_size_that_does_not_divide_evenly()
    test_train_vocabulary_streaming_discriminates_different_scenes()
    print("All VLAD retrieval tests passed.")
