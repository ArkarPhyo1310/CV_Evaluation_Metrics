import re
import setuptools


def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_version():
    with open("cv_evaluation_metrics/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setuptools.setup(
    name="CV_Evaluation_Metrics",
    version=get_version(),
    author="Arkar Phyo",
    url="https://github.com/ArkarPhyo1310/CV_Evaluation_Metrics",
    python_requires=">=3.6",
    install_requires=get_install_requirements(),
    setup_requires=["wheel"],  # avoid building error when pip is not updated
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,  # include files in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    project_urls={
        "Source": "https://github.com/ArkarPhyo1310/CV_Evaluation_Metrics",
        "Tracker": "https://github.com/ArkarPhyo1310/CV_Evaluation_Metrics/issues",
    },
)
