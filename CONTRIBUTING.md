# How to Contribute to the Mortar Analytics Library

Mortar is a community-led effort and will succeed on the contribution of all edits -- big or small -- that seek to improve the quality of the Mortar analytics library.

Following here are some guidelines on how to structure or conduct these edits.

---

### Did you find a bug?

- If you believe the bug is a security vulnerability, please email the Mortar admins @ `mortar-admin [ AT ] protonmail [ DOT ] com`
- Make sure to search the [GitHub Issues](https://github.com/SoftwareDefinedBuildings/mortar-analytics/issues) to see if the bug has already been reported
- If you cannot find an open issue describing the problem, [open a new one](https://github.com/SoftwareDefinedBuildings/mortar-analytics/issues/new). Be sure to include a title and clear description, including as much relevant information as possible. It is crucial to provide a **code sample** or **executable test case** demonstrating the expected behavior that is not occuring.

### Do you want to add to or fix the documentation?

- **Note: the docs will be added in a future repository**
- Open a new GitHub pull request with the changes, after making sure to the best of your ability that it is devoid of errors and mistakes
- ensure that the description of the pull request summarizes what the additions or changes are to the documentation
- if you are unsure about anything or have any questions, please note this in the description
- compile the documentation using [mkdocs](https://www.mkdocs.org/) and check for formatting errors. Do not submit the generated code as part of the pull request

### Do you have a new application?

- Open a new GitHub pull request with the application code
- Unless there is a good reason, Mortar applications should be written to target Python 3.5 or greater (compatible with the `pymortar` library)
- The application should be contained in its own folder and given a proper name
- Application folder should contain an `app.py` file that contains the coordinating logic for the application
- `app.py` should include example usage at the bottom of the file using the following construct:
    ```python
    if __name__ == '__main__':
        # example code for how to use your application goes here
    ```
- the application folder should come with a `requirements.txt` file that enumerates the packages required to execute the application
- the application folder should come with a `README.md` file containing a description of the application
