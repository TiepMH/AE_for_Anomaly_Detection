import matplotlib.pyplot as plt


def plot_a_normal_img_and_its_reconstruction(img_test, img_decoded, angles):
    plt.plot(angles, img_test,
             linestyle='--', linewidth=2, color='k',
             label='(Ground truth) A normal curve from the data')
    plt.plot(angles, img_decoded,
             linestyle='-', linewidth=2, color='r',
             label='(Through DNN) A reconstructed curve')
    # plt.fill_between(angles,
    #                  img_decoded,
    #                  img_test,
    #                  # color='lightcoral',
    #                  label='Area between $2$ curves')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.ylabel(r'$S(\theta)$', fontsize=12)
    plt.xlim(-90, 90)
    plt.ylim(ymin=0)
    plt.tight_layout()
    return None


def plot_an_anomalous_img_and_its_reconstruction(img_test, img_decoded, angles):
    plt.plot(angles, img_test,
             linestyle='--', linewidth=2, color='k',
             label='(Ground truth) An anomalous curve from the data')
    plt.plot(angles, img_decoded,
             linestyle='-', linewidth=2, color='r',
             label='(Through DNN) A reconstructed curve')
    # plt.fill_between(angles,
    #                  img_decoded,
    #                  img_test,
    #                  # color='lightcoral',
    #                  label='Area between $2$ curves')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.ylabel(r'$S(\theta)$', fontsize=12)
    plt.xlim(-90, 90)
    plt.ylim(ymin=0)
    plt.tight_layout()
    return None
