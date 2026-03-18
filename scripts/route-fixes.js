function normalizeRoot(root) {
  if (!root) return '/';
  return root.endsWith('/') ? root : `${root}/`;
}

function joinRoot(root, path) {
  const cleanPath = path.replace(/^\/+/, '');
  return `${normalizeRoot(root)}${cleanPath}`;
}

function createRedirectHtml(targetUrl) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url=${targetUrl}">
  <meta name="robots" content="noindex">
  <link rel="canonical" href="${targetUrl}">
  <script>location.replace(${JSON.stringify(targetUrl)});</script>
  <title>Redirecting...</title>
</head>
<body>
  <p>Redirecting to <a href="${targetUrl}">${targetUrl}</a></p>
</body>
</html>`;
}

hexo.extend.filter.register('after_generate', function() {
  const root = normalizeRoot(hexo.config.root);
  const redirects = {
    'index.html': joinRoot(root, 'home/'),
    'home.html': joinRoot(root, 'home/'),
    'tags.html': joinRoot(root, 'tags/')
  };

  Object.entries(redirects).forEach(([route, target]) => {
    hexo.route.set(route, createRedirectHtml(target));
  });
});
