# Code copied from https://fastapi-users.github.io/fastapi-users/10.1/configuration/full-example/ ; MAKE SURE TO BE ON V13.0!
#### DO NOT CHANGE THIS FILE UNTIL PROJECT WORKS
import uuid

from fastapi_users import schemas
## Side note: If you cannot import the selfwritten modules, this might help, especially when working with venv: https://stackoverflow.com/questions/71754064/vs-code-pylance-problem-with-module-imports
## Search for the installation path of fastapi_users with pip show fastapi_users


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass