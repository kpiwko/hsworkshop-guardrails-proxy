"""Structural checks for the guardrails-proxy GHCR build workflow."""
from __future__ import annotations

import os
import unittest

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_PATH = os.path.join(REPO_ROOT, ".github", "workflows", "build.yaml")
DOCKERFILE_PATH = os.path.join(REPO_ROOT, "Dockerfile")
IMAGE = "ghcr.io/kpiwko/hsworkshop-guardrails-proxy"


class GuardrailsProxyWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW_PATH, encoding="utf-8") as workflow:
            # BaseLoader preserves the GitHub Actions key `on` as a string.
            cls.workflow = yaml.load(workflow, Loader=yaml.BaseLoader)
        with open(DOCKERFILE_PATH, encoding="utf-8") as dockerfile:
            cls.dockerfile = dockerfile.read()

    def test_push_and_manual_triggers_are_on_main(self):
        triggers = self.workflow["on"]
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertEqual(triggers["workflow_dispatch"], "")

    def test_native_build_matrix_uses_the_correct_context_and_permissions(self):
        build = self.workflow["jobs"]["build"]
        matrix = build["strategy"]["matrix"]["include"]
        self.assertEqual(
            matrix,
            [
                {"arch": "amd64", "runner": "ubuntu-24.04"},
                {"arch": "arm64", "runner": "ubuntu-24.04-arm"},
            ],
        )
        self.assertEqual(
            build["permissions"], {"contents": "read", "packages": "write"}
        )
        steps = build["steps"]
        self.assertEqual(steps[0]["uses"], "actions/checkout@v4")
        build_step = next(step for step in steps if step.get("id") == "build")
        self.assertEqual(build_step["uses"], "redhat-actions/buildah-build@v2")
        self.assertEqual(build_step["with"]["context"], ".")
        self.assertEqual(
            build_step["with"]["containerfiles"], "./Dockerfile"
        )
        self.assertEqual(build_step["with"]["platforms"], "linux/${{ matrix.arch }}")
        self.assertIn(
            "org.opencontainers.image.source=https://github.com/kpiwko/hsworkshop-guardrails-proxy",
            build_step["with"]["labels"],
        )
        self.assertIn(
            "org.opencontainers.image.revision=${{ github.sha }}",
            build_step["with"]["labels"],
        )

    def test_native_health_and_arbitrary_uid_smoke_test_precedes_each_arch_push(self):
        steps = self.workflow["jobs"]["build"]["steps"]
        names = [step.get("name") for step in steps]
        smoke = names.index("Smoke-test native image with arbitrary UID")
        push = names.index("Push per-architecture image to GHCR")
        self.assertLess(smoke, push)
        smoke_run = steps[smoke]["run"]
        for expected in (
            "podman image inspect",
            "linux/${ARCH}",
            "--user 123456:0",
            "id -u",
            "/healthz",
            "trap cleanup EXIT",
        ):
            self.assertIn(expected, smoke_run)

    def test_push_uses_approved_ghcr_image_and_github_token(self):
        push = next(
            step for step in self.workflow["jobs"]["build"]["steps"]
            if step.get("name") == "Push per-architecture image to GHCR"
        )
        self.assertEqual(push["uses"], "redhat-actions/push-to-registry@v3")
        self.assertEqual(push["with"]["registry"], "ghcr.io/kpiwko")
        self.assertEqual(push["with"]["image"], "hsworkshop-guardrails-proxy")
        self.assertEqual(push["with"]["username"], "${{ github.actor }}")
        self.assertEqual(push["with"]["password"], "${{ secrets.GITHUB_TOKEN }}")
        self.assertEqual(push["with"]["tags"], "${{ github.sha }}-${{ matrix.arch }}")
        with open(WORKFLOW_PATH, encoding="utf-8") as workflow:
            text = workflow.read()
        self.assertNotIn("secrets.GHCR", text)
        self.assertNotIn("RH_REGISTRY", text)

    def test_manifest_waits_for_both_builds_and_publishes_only_expected_tags(self):
        manifest = self.workflow["jobs"]["manifest"]
        self.assertEqual(manifest["needs"], "build")
        self.assertEqual(manifest["permissions"], {"packages": "write"})
        self.assertNotIn("contents", manifest["permissions"])
        self.assertEqual(
            manifest["steps"][0]["uses"], "docker/login-action@v3"
        )
        create = manifest["steps"][1]["run"]
        self.assertIn("docker buildx imagetools create", create)
        self.assertIn('--tag "${IMAGE}:${GITHUB_SHA}"', create)
        self.assertIn('--tag "${IMAGE}:main"', create)
        self.assertIn('"${IMAGE}:${GITHUB_SHA}-amd64"', create)
        self.assertIn('"${IMAGE}:${GITHUB_SHA}-arm64"', create)
        self.assertIn(IMAGE, manifest["steps"][1]["env"]["IMAGE"])
        self.assertNotIn("latest", create)

    def test_dockerfile_digest_is_pinned_without_forcing_amd64(self):
        self.assertIn(
            "sha256:e695ab189fa2d082b1c61c69a6b888a5adeca61ceeccad107b1ed2a125721a9e",
            self.dockerfile,
        )
        self.assertNotIn("FROM --platform=linux/amd64", self.dockerfile)
        self.assertIn("requirements.lock", self.dockerfile)


if __name__ == "__main__":
    unittest.main()
