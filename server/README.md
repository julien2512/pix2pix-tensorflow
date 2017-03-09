# pix2pix-tensorflow server

Host pix2pix-tensorflow models to be used with something like the [Image-to-Image Demo](https://affinelayer.com/pixsrv/).

This is a simple python server that serves models exported from `pix2pix.py --mode export`.  It can serve local models.

## Local

Using the [julien2512/pix2pix-tensorflow Docker image] from github:

```sh
# export a model to upload

cd server
mkdir models
cd models
wget http://163.172.41.53/volcans_v1.data.tar.gz
gunzip volcans_v1.data.tar.gz
tar xf volcans_v1.data.tar.gz
# run local server
python ../tools/dockrun.py --port 8000 python serve.py --port 8000 --local_models_dir models

If you open [http://localhost:8000/](http://localhost:8000/) in a browser, you should see an interactive demo.

