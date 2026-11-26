const path = require('path');

function getFilenameTitle(source) {
  if (!source) return null;
  const normalized = source.replace(/\\/g, '/');
  if (!normalized.startsWith('_posts/')) return null;
  return path.basename(normalized, path.extname(normalized));
}

// Always sync post title with its filename so renaming a .md file
// automatically updates the displayed title everywhere (home, categories, archives, post page).
hexo.extend.filter.register('before_post_render', function(data) {
  const filenameTitle = getFilenameTitle(data.source);
  if (filenameTitle) {
    data.title = filenameTitle;
  }
  return data;
});

hexo.extend.filter.register('before_generate', function() {
  const Post = hexo.model('Post');
  Post.forEach(post => {
    const filenameTitle = getFilenameTitle(post.source);
    if (filenameTitle && post.title !== filenameTitle) {
      post.title = filenameTitle;
      post.save();
    }
  });
});
