import asyncio
import unittest

from main import health, root


class EndpointTests(unittest.TestCase):
    def test_root(self) -> None:
        response = asyncio.run(root())

        self.assertEqual(
            response,
            {
                "message": "Social Media Privacy Guard API",
                "status": "running",
            },
        )

    def test_health(self) -> None:
        response = asyncio.run(health())

        self.assertEqual(
            response,
            {
                "status": "healthy",
                "service": "privacy-guard-backend",
            },
        )


if __name__ == "__main__":
    unittest.main()
