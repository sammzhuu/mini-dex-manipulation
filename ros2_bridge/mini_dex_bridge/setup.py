from setuptools import find_packages, setup

package_name = "mini_dex_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Samuel Zhu",
    maintainer_email="zjiale1118@gmail.com",
    description="Mini-Dex FDE toolkit ROS2 bridge: capture and policy-command nodes over JSON topics",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "capture_node = mini_dex_bridge.capture_node:main",
            "policy_node = mini_dex_bridge.policy_node:main",
            "replay = mini_dex_bridge.replay:main",
        ],
    },
)
