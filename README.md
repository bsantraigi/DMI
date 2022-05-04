# Discourse-Mutual-Information

**Link to the Hosted Site:** https://bsantraigi.github.io/DMI

## Site backend

- Site was built using jekyll
- Docker images from [bretfisher/jekyll-serve](http://some-link), [bretfisher/jekyll-serve](http://some-link)
```bash
# New Proj
docker run -v $(pwd):/site bretfisher/jekyll new .

# Build -> Serve
docker run -p 4000:4000 -v $(pwd):/site bretfisher/jekyll-serve
```