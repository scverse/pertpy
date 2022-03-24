---
github_url: "{{ fullname }}"
---

{{ fullname | escape | underline}}

```{eval-rst}
.. currentmodule:: {{ module }}
```

```{eval-rst}
.. autoclass:: {{ objname }}


   .. _sphx_glr_backref_{{fullname}}:
```
